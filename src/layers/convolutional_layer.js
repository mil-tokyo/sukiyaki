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
	
	Sukiyaki.Layers.ConvolutionalLayer = function(params) {
		// variables
		this.input_rows = null;
		this.input_cols = null;
		this.input_depth = null;
		this.output_rows = null;
		this.output_cols = null;
		this.output_depth = null;
		this.window_size = null;
		this.padding = null;
		this.w = null;
		this.b = null;
		this.kernel_forward = null;
		
		// constructor
		this.checkContainParams(params, ['input_rows', 'input_cols', 'input_depth', 'output_depth', 'window_size', 'padding']);
		this.input_rows = params.input_rows;
		this.input_cols = params.input_cols;
		this.input_depth = params.input_depth;
		this.output_rows = this.calcOutputSize(params.input_rows, params.window_size, params.padding);
		this.output_cols = this.calcOutputSize(params.input_cols, params.window_size, params.padding);
		this.output_depth = params.output_depth;
		this.window_size = params.window_size;
		this.padding = params.padding;
		this.initW(params.window_size);
		this.initKernel();
		this.input_size = this.input_rows * this.input_cols * this.input_depth;
		this.output_size = this.output_rows * this.output_cols * this.output_depth;
	};
	Sukiyaki.Layers.ConvolutionalLayer.prototype = new Sukiyaki.Layers.LayerBase();
	
	Sukiyaki.Layers.ConvolutionalLayer.prototype.initKernel = function() {
		if (this.window_size <= this.padding) {
			throw new Error('padding must be smaller than window_size');
		}
		if ($M.CL) {
			this.kernel_forward = $M.CL.createKernel([
				"#define I_RS " + this.input_rows,
				"#define O_RS " + this.output_rows,
				"#define I_CS " + this.input_cols,
				"#define O_CS " + this.output_cols,
				"#define I_DS " + this.input_depth,
				"#define O_DS " + this.output_depth,
				"#define W_RS " + this.window_size,
				"#define W_CS " + this.window_size,
				"#define W_PADR " + this.padding,
				"#define W_PADC " + this.padding,
				"#define I_I2D(i) ((i) / (I_RS * I_CS * bs))																						",
				"#define O_I2D(i) ((i) / (O_RS * O_CS * bs))																						",
				"#define I_I2R(i) (((i) / (I_CS * bs)) % I_RS)																						",
				"#define O_I2R(i) (((i) / (O_CS * bs)) % O_RS)																						",
				"#define I_I2C(i) (((i) / bs) % I_CS)																								",
				"#define O_I2C(i) (((i) / bs) % O_CS)																								",
				"#define I2B(i) ((i) % bs)																											",
				"#define I_GET(depth, row, col, batch) (input[(((depth) * I_RS + (row)) * I_CS + (col)) * bs + (batch)])							",
				"#define O_GET(depth, row, col, batch) (output[(((depth) * O_RS + (row)) * O_CS + (col)) * bs + (batch)])							",
				"#define W_GET(input_depth, output_depth, row, col) (w[(((output_depth) * I_DS + (input_depth)) * W_RS + (row)) * W_CS + (col)])	",
				"__kernel void kernel_func(																											",
				"	__global float *output, __global float *input, __global float *w, uint bs, uint iNumElements)									",
				"{																																	",
				"	size_t i =  get_global_id(0);																									",
				"	if(i >= iNumElements) return;																									",
				"	uint o_d = O_I2D(i); uint o_r = O_I2R(i); uint o_c = O_I2C(i); uint b = I2B(i);													",
				"	float tmp = 0.0;																												",
				"	for (uint i_d = 0; i_d < I_DS; i_d++) {																							",
				"		for (uint w_r = 0; w_r < W_RS; w_r++) {																						",
				"			int i_r = o_r + w_r - W_PADR;																							",
				"			if (i_r < 0 || i_r >= I_RS) { continue; }																				",
				"			for (uint w_c = 0; w_c < W_CS; w_c++) {																					",
				"				int i_c = o_c + w_c - W_PADC;																						",
				"				if (i_c < 0 || i_c >= I_CS) { continue; }																			",
				"				tmp += I_GET(i_d, i_r, i_c, b) * W_GET(i_d, o_d, w_r, w_c);															",
				"			}																														",
				"		}																															",
				"	}																																",
				"	output[i] = tmp;																												",
				"}																																	"
			].join('\r\n'));
			
			this.kernel_backward = $M.CL.createKernel([
				"#define I_RS " + this.input_rows,
				"#define O_RS " + this.output_rows,
				"#define I_CS " + this.input_cols,
				"#define O_CS " + this.output_cols,
				"#define I_DS " + this.input_depth,
				"#define O_DS " + this.output_depth,
				"#define W_RS " + this.window_size,
				"#define W_CS " + this.window_size,
				"#define W_PADR " + this.padding,
				"#define W_PADC " + this.padding,
				"#define I_I2D(i) ((i) / (I_RS * I_CS * bs))																												",
				"#define O_I2D(i) ((i) / (O_RS * O_CS * bs))																												",
				"#define I_I2R(i) (((i) / (I_CS * bs)) % I_RS)																												",
				"#define O_I2R(i) (((i) / (O_CS * bs)) % O_RS)																												",
				"#define I_I2C(i) (((i) / bs) % I_CS)																														",
				"#define O_I2C(i) (((i) / bs) % O_CS)																														",
				"#define I2B(i) ((i) % bs)																																	",
				"#define I_GET(depth, row, col, batch) (input[(((depth) * I_RS + (row)) * I_CS + (col)) * bs + (batch)])													",
				"#define O_GET(depth, row, col, batch) (output[(((depth) * O_RS + (row)) * O_CS + (col)) * bs + (batch)])													",
				"#define W_GET(input_depth, output_depth, row, col) (w[(((output_depth) * I_DS + (input_depth)) * W_RS + (row)) * W_CS + (col)])							",
				"#define W_ROTATE_GET(input_depth, output_depth, row, col) (w[(((output_depth) * I_DS + (input_depth)) * W_RS + (W_RS-1-(row))) * W_CS + (W_CS-1-(col))])	",
				"__kernel void kernel_func(																																	",
				"	__global float *output, __global float *input, __global float *w, uint bs, uint iNumElements)															",
				"{																																							",
				"	size_t i =  get_global_id(0);																															",
				"	if(i >= iNumElements) return;																															",
				"	uint i_d = I_I2D(i); uint i_r = I_I2R(i); uint i_c = I_I2C(i); uint b = I2B(i);																			",
				"	float tmp = 0.0;																																		",
				"	for (uint o_d = 0; o_d < O_DS; o_d++) {																													",
				"		for (uint w_r = 0; w_r < W_RS; w_r++) {																												",
				"			int o_r = i_r + w_r - (W_RS - 1) + W_PADR;																										",
				"			if (o_r < 0 || o_r >= O_RS) { continue; }																										",
				"			for (uint w_c = 0; w_c < W_CS; w_c++) {																											",
				"				int o_c = i_c + w_c - (W_CS - 1) + W_PADC;																									",
				"				if (o_c < 0 || o_c >= O_CS) { continue; }																									",
				"				tmp += O_GET(o_d, o_r, o_c, b) * W_ROTATE_GET(i_d, o_d, w_r, w_c);																			",
				"			}																																				",
				"		}																																					",
				"	}																																						",
				"	input[i] = tmp;																																			",
				"}																																							"
			].join('\r\n'));
			
			this.kernel_update = $M.CL.createKernel([
				"#define I_RS " + this.input_rows,
				"#define O_RS " + this.output_rows,
				"#define I_CS " + this.input_cols,
				"#define O_CS " + this.output_cols,
				"#define I_DS " + this.input_depth,
				"#define O_DS " + this.output_depth,
				"#define W_RS " + this.window_size,
				"#define W_CS " + this.window_size,
				"#define W_PADR " + this.padding,
				"#define W_PADC " + this.padding,
				"#define I_I2D(i) ((i) / (I_RS * I_CS * bs))																												",
				"#define O_I2D(i) ((i) / (O_RS * O_CS * bs))																												",
				"#define I_I2R(i) (((i) / (I_CS * bs)) % I_RS)																												",
				"#define O_I2R(i) (((i) / (O_CS * bs)) % O_RS)																												",
				"#define I_I2C(i) (((i) / bs) % I_CS)																														",
				"#define O_I2C(i) (((i) / bs) % O_CS)																														",
				"#define W_I2OD(i) ((i) / (W_RS * W_CS * I_DS))																												",
				"#define W_I2ID(i) (((i) / (W_RS * W_CS)) % I_DS)																											",
				"#define W_I2R(i) (((i) / W_CS) % W_RS)																														",
				"#define W_I2C(i) ((i) % W_CS)																																",
				"#define I2B(i) ((i) % bs)																																	",
				"#define I_GET(depth, row, col, batch) (input[(((depth) * I_RS + (row)) * I_CS + (col)) * bs + (batch)])													",
				"#define I_ROTATE_GET(depth, row, col, batch) (input[(((depth) * I_RS + (I_RS-1-(row))) * I_CS + (I_CS-1-(col))) * bs + (batch)])							",
				"#define O_GET(depth, row, col, batch) (output[(((depth) * O_RS + (row)) * O_CS + (col)) * bs + (batch)])													",
				"__kernel void kernel_func(																																	",
				"	__global float *output, __global float *input, __global float *w, uint bs, uint iNumElements)												",
				"{																																							",
				"	size_t i =  get_global_id(0);																															",
				"	if(i >= iNumElements) return;																															",
				"	uint w_od = W_I2OD(i); uint w_id = W_I2ID(i); uint w_r = W_I2R(i); uint w_c = W_I2C(i); 																",
				"	float tmp = 0.0;																																		",
				"	for (uint b = 0; b < bs; b++) {																															",
				"		for (uint o_r = 0; o_r < O_RS; o_r++) {																												",
				"			int i_r = w_r + o_r - W_PADR;																													",
				"			if (i_r < 0 || i_r >= I_RS) { continue; }																										",
				"			for (uint o_c = 0; o_c < O_CS; o_c++) {																											",
				"				int i_c = w_c + o_c - W_PADC;																												",
				"				if (i_c < 0 || i_c >= I_CS) { continue; }																									",
				"				tmp += I_GET(w_id, i_r, i_c, b) * O_GET(w_od, o_r, o_c, b);																					",
				"			}																																				",
				"		}																																					",
				"	}																																						",
				"	w[i] = tmp / (float)bs;																																	",
				"}																																							"
			].join('\r\n'));
		};
	};
	
	Sukiyaki.Layers.ConvolutionalLayer.prototype.calcOutputSize = function(input, window_size, padding) {
		return input - window_size + 1 + padding * 2;
	};
	
	Sukiyaki.Layers.ConvolutionalLayer.prototype.initW = function(window_size) {
		// w and b
		var std = Math.sqrt(1.0 / (window_size * window_size * this.input_depth));
		this.w = new $M(this.output_depth * this.input_depth, this.window_size * this.window_size);
		this.w.gaussRandom(0, std);
		this.b = new $M(this.output_rows * this.output_cols * this.output_depth, 1);
		this.b.largeZeros();
		// for AdaGrad
		this.adagrad_w = new $M(this.w.rows, this.w.cols);
		this.adagrad_w.largeZeros(0.01);
		this.adagrad_b = new $M(this.b.rows, this.b.cols);
		this.adagrad_b.largeZeros(0.01);
	};
	
	Sukiyaki.Layers.ConvolutionalLayer.prototype.calculateUpdateParams = function() {
		this.delta_w = new $M(this.w.rows, this.w.cols);
		if ($M.CL) {
			$M.CL.executeKernel(
				this.kernel_update,
				[
					{ access : WebCL.MEM_READ_ONLY, datum : this.delta_output },
					{ access : WebCL.MEM_READ_ONLY, datum : this.input },
					{ access : WebCL.MEM_WRITE, datum : this.delta_w },
					{ datum : this.input.cols, type : WebCL.type.UINT },
					{ datum : this.delta_w.length, type : WebCL.type.UINT }
				],
				this.delta_w.length
			);
		} else {
			var o_ds = this.output_depth;
			var o_rs = this.output_rows;
			var o_cs = this.output_cols;
			var bs = this.input.cols;
			var i_ds = this.input_depth;
			var i_rs = this.input_rows;
			var i_cs = this.input_cols;
			var w_rs = this.window_size;
			var w_cs = this.window_size;
			var w_padr = this.padding;
			var w_padc = this.padding;
			this.input.syncData();
			this.delta_output.syncData();
			this.delta_w.syncData();
			var input = this.input.data;
			var output = this.delta_output.data;
			var w = this.delta_w.data;
			for (var w_od = 0; w_od < o_ds; w_od++) {
				for (var w_id = 0; w_id < i_ds; w_id++) {
					for (var w_r = 0; w_r < w_rs; w_r++) {
						for (var w_c = 0; w_c < w_cs; w_c++) {
							var tmp = 0;
							for (var o_r = 0; o_r < o_rs; o_r++) {
								var i_r = w_r + o_r - w_padr;
								if (i_r < 0 || i_r >= i_rs) { continue; }
								for (var o_c = 0; o_c < o_cs; o_c++) {
									var i_c = w_c + o_c - w_padc;
									if (i_c < 0 || i_c >= i_cs) { continue; }
									for (var b = 0; b < bs; b++) {
										tmp += input[((w_id * i_rs + i_r) * i_cs + i_c) * bs + b] * output[((w_od * o_rs + o_r) * o_cs + o_c) * bs + b];
									}
								}
							}
							w[((w_od * i_ds + w_id) * w_rs + w_r) * w_cs + w_c] = tmp / bs;
						}
					}
				}
			}
		}
		
		this.delta_b = $M.largeSumEachRow(this.delta_output).largeTimes(1.0 / this.input.cols);
	};
	
	Sukiyaki.Layers.ConvolutionalLayer.prototype.update = function() {
		var sqrt_each = $M.CL ? $M.CL.mapGenerator('sqrt(a[i])') : function(xs) { return xs.map(Math.sqrt); };
		return function() {
			if (this.use_adagrad) {
				this.calcAdaGrad();
				var w_alpha = this.adagrad_w.largeClone();
				sqrt_each(w_alpha);
				var b_alpha = this.adagrad_b.largeClone();
				sqrt_each(b_alpha);
				this.w.largeSub(this.delta_w.largeDivEach(w_alpha).largeTimes(this.learn_rate));
				this.b.largeSub(this.delta_b.largeDivEach(b_alpha).largeTimes(this.learn_rate));
				w_alpha.destruct();
				b_alpha.destruct();
			} else {
				this.w.largeSub(this.delta_w.largeTimes(this.learn_rate));
				this.b.largeSub(this.delta_b.largeTimes(this.learn_rate));
			}
			this.delta_w.destruct();
			this.delta_b.destruct();
		};
	}();
	
	Sukiyaki.Layers.ConvolutionalLayer.prototype.calcAdaGrad = function() {
		var square_each = $M.CL ? $M.CL.mapGenerator('a[i] * a[i]') : function(xs) { return xs.map(function(x) { return x * x }); };
		return function() {
			var square_delta_w = this.delta_w.largeClone();
			square_each(square_delta_w);
			this.adagrad_w.largeAdd(square_delta_w);
			square_delta_w.destruct();
			var square_delta_b = this.delta_b.largeClone();
			square_each(square_delta_b);
			this.adagrad_b.largeAdd(square_delta_b);
			square_delta_b.destruct();
		};
	}();
	
	Sukiyaki.Layers.ConvolutionalLayer.prototype.forward = function() {
		if (!this.input.row_wise) {
			throw new Error('forward method for col wise matrix is not implemented yet');
		}
		if (this.input.rows !== this.input_size) {
			throw new Error('input size does not match');
		}
		this.output = new $M(this.output_rows * this.output_cols * this.output_depth, this.input.cols);
		if ($M.CL) {
			$M.CL.executeKernel(
				this.kernel_forward,
				[
					{ access : WebCL.MEM_WRITE_ONLY, datum : this.output },
					{ access : WebCL.MEM_READ_ONLY, datum : this.input },
					{ access : WebCL.MEM_READ_ONLY, datum : this.w },
					{ datum : this.input.cols, type : WebCL.type.UINT },
					{ datum : this.output.length, type : WebCL.type.UINT }
				],
				this.output.length
			);
		} else {
			var o_ds = this.output_depth;
			var o_rs = this.output_rows;
			var o_cs = this.output_cols;
			var bs = this.input.cols;
			var i_ds = this.input_depth;
			var i_rs = this.input_rows;
			var i_cs = this.input_cols;
			var w_rs = this.window_size;
			var w_cs = this.window_size;
			var w_padr = this.padding;
			var w_padc = this.padding;
			this.input.syncData();
			this.output.syncData();
			this.w.syncData();
			var input = this.input.data;
			var output = this.output.data;
			var w = this.w.data;
			for (var o_d = 0; o_d < o_ds; o_d++) {
				for (var o_r = 0; o_r < o_rs; o_r++) {
					for (var o_c = 0; o_c < o_cs; o_c++) {
						for (var b = 0; b < bs; b++) {
							var tmp = 0;
							for (var i_d = 0; i_d < i_ds; i_d++) {
								for (var w_r = 0; w_r < w_rs; w_r++) {
									var i_r = o_r + w_r - w_padr;
									if (i_r < 0 || i_r >= i_rs) { continue; }
									for (var w_c = 0; w_c < w_cs; w_c++) {
										var i_c = o_c + w_c - w_padc;
										if (i_c < 0 || i_c >= i_cs) { continue; }
										tmp += input[((i_d * i_rs + i_r) * i_cs + i_c) * bs + b] * w[((o_d * i_ds + i_d) * w_rs + w_r) * w_cs + w_c];
									}
								}
							}
							output[((o_d * o_rs + o_r) * o_cs + o_c) * bs + b] = tmp;
						}
					}
				}
			}
		}
		this.output.largeAdd(this.b);
	};
	
	Sukiyaki.Layers.ConvolutionalLayer.prototype.backward = function() {
		if (!this.delta_output.row_wise) {
			throw new Error('backward method for col wise matrix is not implemented yet');
		}
		if (this.delta_output.rows !== this.output_size) {
			throw new Error('output size does not match');
		}
		this.delta_input = new $M(this.input_rows * this.input_cols * this.input_depth, this.input.cols);
		if ($M.CL) {
			$M.CL.executeKernel(
					this.kernel_backward,
					[
					{ access : WebCL.MEM_READ_ONLY, datum : this.delta_output },
					{ access : WebCL.MEM_WRITE_ONLY, datum : this.delta_input },
					{ access : WebCL.MEM_READ_ONLY, datum : this.w },
					{ datum : this.input.cols, type : WebCL.type.UINT },
					{ datum : this.delta_input.length, type : WebCL.type.UINT }
				],
				this.delta_input.length
			);
		} else {
			var o_ds = this.output_depth;
			var o_rs = this.output_rows;
			var o_cs = this.output_cols;
			var bs = this.input.cols;
			var i_ds = this.input_depth;
			var i_rs = this.input_rows;
			var i_cs = this.input_cols;
			var w_rs = this.window_size;
			var w_cs = this.window_size;
			var w_padr = this.padding;
			var w_padc = this.padding;
			this.delta_input.syncData();
			this.delta_output.syncData();
			this.w.syncData();
			var input = this.delta_input.data;
			var output = this.delta_output.data;
			var w = this.w.data;
			for (var i_d = 0; i_d < i_ds; i_d++) {
				for (var i_r = 0; i_r < i_rs; i_r++) {
					for (var i_c = 0; i_c < i_cs; i_c++) {
						for (var b = 0; b < bs; b++) {
							var tmp = 0;
							for (var o_d = 0; o_d < o_ds; o_d++) {
								for (var w_r = 0; w_r < w_rs; w_r++) {
									var o_r = i_r + w_r - (w_rs - 1) + w_padr;
									if (o_r < 0 || o_r >= o_rs) { continue; }
									for (var w_c = 0; w_c < w_cs; w_c++) {
										var o_c = i_c + w_c - (w_cs - 1) + w_padc;
										if (o_c < 0 || o_c >= o_cs) { continue; }
										tmp += output[((o_d * o_rs + o_r) * o_cs + o_c) * bs + b] * w[((o_d * i_ds + i_d) * w_rs + (w_rs-1-w_r)) * w_cs + (w_cs-1-w_c)];
									}
								}
							}
							input[((i_d * i_rs + i_r) * i_cs + i_c) * bs + b] = tmp;
						}
					}
				}
			}
		}
	};
	
	Sukiyaki.Layers.ConvolutionalLayer.prototype.destruct = function() {
		this.w.destruct();
		this.b.destruct();
	};
	
	Sukiyaki.Layers.ConvolutionalLayer.prototype.saveToJson = function() {
		this.w.syncData();
		this.b.syncData();
		return {
			w : this.w.data.buffer.toBase64(),
			b : this.b.data.buffer.toBase64()
		};
	};
	Sukiyaki.Layers.ConvolutionalLayer.prototype.loadFromJson = function(data) {
		this.w.syncData();
		this.b.syncData();
		this.w.data.buffer.fromBase64(data.w);
		this.b.data.buffer.fromBase64(data.b);
	};
})(typeof window === 'undefined', Sushi.Matrix);