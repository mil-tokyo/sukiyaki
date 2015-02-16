(function (nodejs, $M) {
	if (nodejs) {
		require('./layer_base');
	}
	
	/*
	 * input matrix is composed like this (row, col, depth, batch)
	 * 
	 * (0, 0, 0, 0) (0, 0, 0, 1), (0, 0, 0, 2)...
	 * (0, 1, 0, 0) (0, 1, 0, 1), (0, 1, 0, 2)...
	 * (0, 2, 0, 0)...
	 * ...
	 * (0, cols, 0, 0)...
	 * (1, 0, 0, 0)...
	 * (1, 1, 0, 0)...
	 * ...
	 * (rows, cols, 0, 0)...
	 * (0, 0, 1, 0)...
	 * (0, 1, 1, 0)...
	 * ...
	 * 
	 * depth : i / (rows * cols * batches)
	 * row : (i / (cols * batches)) % rows
	 * col : (i / batches) % cols
	 * batch : i % batches
	 * 
	 * i : ((depth * rows + row) * cols + col) * batches + batch
	 */

	Sukiyaki.Layers.MaxPoolingLayer = function(params) {
		// variables
		this.kernel_forward = null;
		this.kernel_backward = null;
		this.map = null;
		this.input_rows = null;
		this.input_cols = null;
		this.input_depth = null;
		this.output_rows = null;
		this.output_cols = null;
		this.output_depth = null;
		this.window_size = null;
		this.stride = null;
		
		// constructor
		this.checkContainParams(params, ['input_rows', 'input_cols', 'input_depth', 'window_size', 'stride']);
		this.input_rows = params.input_rows;
		this.input_cols = params.input_cols;
		this.input_depth = params.input_depth;
		this.output_rows = this.calcPooledSize(params.input_rows, params.window_size, params.stride);
		this.output_cols = this.calcPooledSize(params.input_cols, params.window_size, params.stride);
		this.output_depth = params.input_depth;
		this.initKernel(params.input_rows, params.input_cols, this.output_rows, this.output_cols, params.input_depth, params.window_size, params.stride);
		this.input_size = params.input_rows * params.input_cols * params.input_depth;
		this.output_size = this.output_rows * this.output_cols * this.output_depth;
		this.window_size = params.window_size;
		this.stride = params.stride;
	};

	Sukiyaki.Layers.MaxPoolingLayer.prototype = new Sukiyaki.Layers.LayerBase();

	Sukiyaki.Layers.MaxPoolingLayer.prototype.initKernel = function(input_rows, input_cols, output_rows, output_cols, depth, window_size, stride) {
		if ($M.CL) {
			this.kernel_forward = $M.CL.createKernel([
				"#define OUTPUT_ROWS_x_OUTPUT_COLS " + (output_rows * output_cols),
				"#define OUTPUT_ROWS " + output_rows,
				"#define OUTPUT_COLS " + output_cols,
				"#define STRIDE " + stride,
				"#define INPUT_ROWS " + input_rows,
				"#define INPUT_COLS " + input_cols,
				"#define WINDOW_SIZE " + window_size,
				"__kernel void kernel_func(__global float *output, __global float *input, __global float *map, uint batches, uint iNumElements)		",
				"{																																	",
				"    size_t i =  get_global_id(0);																									",
				"    if(i >= iNumElements) return;																									",
				"    uint depth = i / (OUTPUT_ROWS_x_OUTPUT_COLS * batches);																		",
				"    uint row = ((i / (OUTPUT_COLS * batches)) % OUTPUT_ROWS) * STRIDE;																",
				"    uint col = ((i / batches) % OUTPUT_COLS) * STRIDE;																				",
				"    uint batch = i % batches;																										",
				"    uint search = ((depth * INPUT_ROWS + row) * INPUT_COLS + col) * batches + batch;												",
				"    output[i] = input[search];																										",
				"    map[i] = search;																												",
				"    for (uint x = 0; x < WINDOW_SIZE; x++) {																						",
				"        for (uint y = 0; y < WINDOW_SIZE; y++) {																					",
				"            search = ((depth * INPUT_ROWS + row + y) * INPUT_COLS + col + x) * batches + batch;									",
				"            if (input[search] > output[i]) {																						",
				"                output[i] = input[search];																							",
				"                map[i] = search;																									",
				"            }																														",
				"        }																															",
				"    }																																",
				"}																																	"
				].join('\r\n')
			);
			this.kernel_backward = $M.CL.createKernel([
					"__kernel void kernel_func(__global float *output, __global float *input, __global float *map, uint batches, uint rows, uint iNumElements)   ",
					"{                                                                                            ",
					"    size_t i =  get_global_id(0);                                                            ",
					"    if(i >= iNumElements) return;                                                            ",
					"    for (uint j = i; j < batches * rows; j+= batches) {                                      ",
					"        input[(uint)map[j]] += output[j];                                                    ",
					"    }                                                                                        ",
					"}                                                                                            "].join('\r\n')
				);
		}
	};

	Sukiyaki.Layers.MaxPoolingLayer.prototype.calcPooledSize = function(input, window_size, stride) {
		return Math.floor((input - window_size) / stride) + 1;
	};

	Sukiyaki.Layers.MaxPoolingLayer.prototype.update = function() { };

	Sukiyaki.Layers.MaxPoolingLayer.prototype.calculateUpdateParams = function() { };

	Sukiyaki.Layers.MaxPoolingLayer.prototype.forward = function() {
		if (!this.input.row_wise) {
			throw new Error('forward method for col wise matrix is not implemented yet');
		}
		if (this.input.rows !== this.input_size) {
			throw new Error('input size does not match');
		}
		this.map = new $M(this.output_size, this.input.cols);
		this.output = new $M(this.output_size, this.input.cols);
		if ($M.CL) {
			$M.CL.executeKernel(
				this.kernel_forward,
				[
					{ access : WebCL.MEM_WRITE_ONLY, datum : this.output },
					{ access : WebCL.MEM_READ_ONLY, datum : this.input },
					{ access : WebCL.MEM_WRITE_ONLY, datum : this.map },
					{ datum : this.input.cols, type : WebCL.type.UINT},
					{ datum : this.output.length, type : WebCL.type.UINT }
				],
				this.output.length
			);
		} else {
			var o_ds = this.output_depth;
			var o_rs = this.output_rows;
			var o_cs = this.output_cols;
			var i_ds = this.input_depth;
			var i_rs = this.input_rows;
			var i_cs = this.input_cols;
			var window_size = this.window_size;
			var stride = this.stride;
			var bs = this.input.cols;
			this.input.syncData();
			this.output.syncData();
			this.map.syncData();
			var input = this.input.data;
			var output = this.output.data;
			var map = this.map.data;
			for (var o_d = 0; o_d < o_ds; o_d++) {
				for (var o_r = 0; o_r < o_rs; o_r++) {
					for (var o_c = 0; o_c < o_cs; o_c++) {
						for (var b = 0; b < bs; b++) {
							var origin = ((o_d * i_rs + o_r * stride) * i_cs + o_c * stride) * bs + b;
							var tmp_max = input[origin];
							var tmp_max_search = origin;
							for (var y = 0; y < window_size; y++) {
								for (var x = 0; x < window_size; x++) {
									var search = origin + (y * i_cs + x) * bs;
									if (input[search] > tmp_max) {
										tmp_max = input[search];
										tmp_max_search = search;
									}
								}
							}
							var o_idx = ((o_d * o_rs + o_r) * o_cs + o_c) * bs + b;
							output[o_idx] = tmp_max;
							map[o_idx] = tmp_max_search;
						}
					}
				}
			}
		}
	};

	Sukiyaki.Layers.MaxPoolingLayer.prototype.backward = function() {
		if (!this.delta_output.row_wise) {
			throw new Error('backward method for col wise matrix is not implemented yet');
		}
		if (this.delta_output.rows !== this.output_size) {
			throw new Error('output size does not match');
		}
		this.delta_input = new $M(this.input.rows, this.input.cols);
		this.delta_input.largeZeros();
		if ($M.CL) {
			$M.CL.executeKernel(
				this.kernel_backward,
				[
					{ access : WebCL.MEM_READ_ONLY, datum : this.delta_output },
					{ access : WebCL.MEM_READ_WRITE, datum : this.delta_input }, // to make sure all elements are zero
					{ access : WebCL.MEM_READ_ONLY, datum : this.map },
					{ datum : this.map.cols, type : WebCL.type.UINT},
					{ datum : this.map.rows, type : WebCL.type.UINT},
					{ datum : this.delta_output.cols, type : WebCL.type.UINT }
				],
				this.delta_output.cols
			);
		} else {
			var o_ds = this.output_depth;
			var o_rs = this.output_rows;
			var o_cs = this.output_cols;
			var bs = this.input.cols;
			this.delta_input.syncData();
			this.delta_output.syncData();
			this.map.syncData();
			var input = this.delta_input.data;
			var output = this.delta_output.data;
			var map = this.map.data;
			for (var o_d = 0; o_d < o_ds; o_d++) {
				for (var o_r = 0; o_r < o_rs; o_r++) {
					for (var o_c = 0; o_c < o_cs; o_c++) {
						for (var b = 0; b < bs; b++) {
							input[
								map[(((o_d * o_rs) + o_r) * o_cs + o_c) * bs + b]
							] += output[(((o_d * o_rs) + o_r) * o_cs + o_c) * bs + b];
						}
					}
				}
			}
		}
	};

	Sukiyaki.Layers.MaxPoolingLayer.prototype.release = function() {
		if (this.map) { 
			this.map.destruct();
		}
	};

	Sukiyaki.Layers.MaxPoolingLayer.prototype.saveToJson = function() { return { }; };
	Sukiyaki.Layers.MaxPoolingLayer.prototype.loadFromJson = function(data) { };
})(typeof window === 'undefined', Sushi.Matrix);