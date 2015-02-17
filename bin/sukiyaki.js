/* begin : sukiyaki.js */
(function (nodejs, $M) {
	if (Sukiyaki) {
		return;
	}
	
	var Sukiyaki = function(layer_params, use_adagrad) {
		// variables
		this.layers = null;
		this.notifier = function() {};
		this.fitting_error = null;
		this.fitting_error_count = null;
		this.layer_params = null;
		if (use_adagrad === void 0) {
			use_adagrad = true;
		}
		this.use_adagrad = use_adagrad;
		
		// constructor
		this.layers = this.parseLayerParams(layer_params);
		if ($M.CL) {
			this.fitting_error = new $M(1, 1);
			this.fitting_error.largeZeros();
		} else {
			this.fitting_error = 0.0;
		}
		this.fitting_error_count = 0;
	};
	
	Sukiyaki.prototype = {
		parseLayerParams : function(layer_params) {
			this.layer_params = layer_params;
			var layer_classes = {
				conv : Sukiyaki.Layers.ConvolutionalLayer,
				pool : Sukiyaki.Layers.MaxPoolingLayer,
				fc : Sukiyaki.Layers.FullyConnectedLayer,
				act : Sukiyaki.Layers.ActivateLayer
			};
			var layers = [];
			for (var i = 0; i < layer_params.length; i++) {
				var layer = new (layer_classes[layer_params[i].type])(layer_params[i].params);
				layer.use_adagrad = this.use_adagrad;
				layers.push(layer);
			}
			return layers;
		},
		saveToJson : function() {
			var layer_status = [];
			for (var i = 0; i < this.layers.length; i++) {
				layer_status.push(this.layers[i].saveToJson());
			}
			return {
				layer_params : this.layer_params,
				layer_status : layer_status
			};
		},
		predict : function(test_batch) {
			var outputs = this.forward(test_batch);
			var predicts = $M.largeArgmaxEachCol(outputs);
			outputs.destruct();
			this.release();
			return predicts;
		},
		learn : function(train_batches, from, to, return_error) {
			var train_batch;
			for (var i = from; i < to; i++) {
				train_batch = train_batches[i];
				this.forward(train_batch.input);
				this.backward(train_batch.output);
				this.update();
				if (i % 100 === 99) {
					this.notifier({
						type : 'learn_batches_finished_count',
						params : {
							batch_count : i + 1,
							batch_all : train_batches.length
						}
					});
				}
				if ($M.CL) {
					var squareSum = this.calcSquareSum(this.layers[this.layers.length - 1].delta_output);
					this.fitting_error.largeAdd(squareSum.largeTimes(1.0 / this.layers[this.layers.length - 1].delta_output.cols));
					squareSum.destruct();
				} else {
					this.fitting_error += this.calcSquareSum(this.layers[this.layers.length - 1].delta_output) / this.layers[this.layers.length - 1].delta_output.cols;
				}
				if (i < to - 1) {
					this.release();
				}
				this.fitting_error_count++;
			}
			var ret_val = void 0;
			if (return_error) {
				if ($M.CL) {
					ret_val = this.fitting_error.get(0, 0) / this.fitting_error_count;
					this.fitting_error.set(0, 0, 0);
				} else {
					ret_val = this.fitting_error / this.fitting_error_count;
					this.fitting_error = 0.0;
				}
				this.fitting_error_count = 0;
			}
			this.release();
			return ret_val;
		},
		setNotifier : function(notifier) {
			this.notifier = notifier;
		},
		calcSquareSum : function() {
			if ($M.CL) {
				var square = $M.CL.mapGenerator('a[i] * a[i]');
				return function(xs) {
					square(xs);
					var row_sum = $M.CL.sumEachRow(xs);
					var col_sum = $M.CL.sumEachCol(row_sum);
					row_sum.destruct();
					return col_sum;
				};
			} else {
				return function(xs) {
					xs.map(function(x) { return x * x });
					return $M.largeSum(xs);
				};
			}
		}(),
		forward : function(input) {
			this.layers[0].input = input;
			for (var i = 0; i < this.layers.length; i++) {
				this.layers[i].forward();
				if (i !== this.layers.length - 1) {
					this.layers[i+1].input = this.layers[i].output;
				}
			}
			return this.layers[this.layers.length - 1].output;
		},
		backward : function(correct_output) {
			this.layers[this.layers.length-1].delta_output = $M.largeSub(this.layers[this.layers.length-1].output, correct_output);
			this.layers[this.layers.length-1].backward();
			for (var i = this.layers.length - 2; i >= 0; i--) {
				this.layers[i].delta_output = this.layers[i+1].delta_input;
				if (i > 0) {
					this.layers[i].backward();
				}
			}
		},
		update : function() {
			for (var i = 0; i < this.layers.length; i++) {
				this.layers[i].calculateUpdateParams();
				this.layers[i].update();
			}
		},
		release : function() {
			for (var i = 0; i < this.layers.length; i++) {
				if (this.layers[i].delta_output) {
					this.layers[i].delta_output.destruct();
				}
				if (this.layers[i].output) {
					this.layers[i].output.destruct();
				}
				this.layers[i].release();
			}
		},
		destruct : function() {
			for (var i = 0; i < this.layers.length; i++) {
				this.layers[i].destruct();
			}
		}
	};
	
	Sukiyaki.loadFromJson = function(data) {
		var sukiyaki = new Sukiyaki(data.layer_params);
		for (var i = 0; i < sukiyaki.layers.length; i++) {
			sukiyaki.layers[i].loadFromJson(data.layer_status[i]);
		}
		return sukiyaki;
	};
	
	(('global', eval)('this')).Sukiyaki = Sukiyaki;
})(typeof window === 'undefined', Sushi.Matrix);

/* end : sukiyaki.js */

/* begin : layers/layer_base.js */
(function (nodejs, $M) {
	Sukiyaki.Layers = Sukiyaki.Layers || { };
	Sukiyaki.Layers.LayerBase = function() {
		// variables
		this.input = null;
		this.output = null;
		this.delta_input = null;
		this.delta_output = null;
		this.learn_rate = 0.05;
		this.input_size = null;
		this.output_size = null;
		this.use_adagrad = true;
	};
	
	Sukiyaki.Layers.LayerBase.prototype = {
		update : function() { throw new Error('not implemented'); return false; },
		calculateUpdateParams : function() { throw new Error('not implemented'); return false; },
		forward : function() { throw new Error('not implemented'); return false; },
		backward : function() { throw new Error('not implemented'); return false; },
		release : function() { },
		destruct : function() { },
		checkContainParams : function(param, requisitions) {
			for (var i = 0; i < requisitions.length; i++) {
				if (param[requisitions[i]] === void 0) {
					throw new Error('Parameter : ' + requisitions[i] + ' is not defined');
				}
			}
		},
		saveToJson : function() { throw new Error('not implemented'); return false; },
		loadFromJson : function(data) { throw new Error('not implemented'); return false; }
	};
	
	ArrayBuffer.prototype.toBase64 = function() {
		var bytes = new Uint8Array(this);
		if (nodejs) {
			return (new Buffer(bytes)).toString("base64");
		} else {
			var binary = '';
			var len = bytes.byteLength;
			for (var i = 0; i < len; i++) {
				binary += String.fromCharCode(bytes[i]);
			}
			return window.btoa(binary);
		}
	};

	ArrayBuffer.prototype.fromBase64 = function(base64) {
		var bytes = new Uint8Array(this);
		if (nodejs) {
			var chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
			var bufferLength = base64.length * 0.75, len = base64.length, i, p = 0, encoded1, encoded2, encoded3, encoded4;
			if (base64[base64.length - 1] === "=") {
				bufferLength--;
				if (base64[base64.length - 2] === "=") {
					bufferLength--;
				}
			}
			if (bufferLength !== this.byteLength) {
				throw new Error('length does not match');
			}

			for (i = 0; i < len; i += 4) {
				encoded1 = chars.indexOf(base64[i]);
				encoded2 = chars.indexOf(base64[i + 1]);
				encoded3 = chars.indexOf(base64[i + 2]);
				encoded4 = chars.indexOf(base64[i + 3]);

				bytes[p++] = (encoded1 << 2) | (encoded2 >> 4);
				bytes[p++] = ((encoded2 & 15) << 4) | (encoded3 >> 2);
				bytes[p++] = ((encoded3 & 3) << 6) | (encoded4 & 63);
			}
		} else {
			var binary = window.atob(base64);
			var len = binary.length;
			if (len !== this.byteLength) {
				throw new Error('length does not match');
			}
			for (var i = 0; i < len; i++) {
				bytes[i] = binary.charCodeAt(i);
			}
		}
	};
})(typeof window === 'undefined', Sushi.Matrix);
/* end : layers/layer_base.js */

/* begin : layers/activate_layer.js */
(function (nodejs, $M) {
	if (nodejs) {
		
	}
	
	Sukiyaki.Layers.ActivateLayer = function(params) {
		// variables
		this.activate_function = function(xs) { };
		this.activate_function_div = function(xs) { };
		
		// constructor
		this.checkContainParams(params, ['activation_type']);
		this.setActivationType(params.activation_type);
	};
	
	Sukiyaki.Layers.ActivateLayer.prototype = new Sukiyaki.Layers.LayerBase();

	Sukiyaki.Layers.ActivateLayer.prototype.setActivationType = function(name) {
		switch (name) {
			case 'sigmoid':
				if ($M.CL) {
					this.activate_function = function() {
						var sigmoid = $M.CL.mapGenerator('1.0 / (exp(-a[i]) + 1.0)');
						return function(xs) {
							sigmoid(xs);
							return xs;
						};
					}();
					this.activate_function_div = function() {
						var sigmoid_div = $M.CL.mapGenerator('a[i] * (1.0 - a[i])');
						return function(xs) {
							sigmoid_div(xs);
							return xs;
						};
					}();
				} else {
					this.activate_function = function(xs) { return xs.map(function(x) { return 1.0 / (Math.exp(-x) + 1.0) }); };
					this.activate_function_div = function(xs) { return xs.map(function(x) { return x * (1.0 - x); }); };
				}
				break;
			case 'softmax':
				if ($M.CL) {
					this.activate_function = function() {
						var exp = $M.CL.mapGenerator('exp(a[i])');
						return function(xs) {
							var maxEachCol = $M.largeMaxEachCol(xs);
							xs.largeSub(maxEachCol);
							maxEachCol.destruct();
							exp(xs);
							var sumEachCol = $M.largeSumEachCol(xs);
							xs.largeDivEach(sumEachCol);
							sumEachCol.destruct();
							return xs;
						};
					}();
					this.activate_function_div = function() {
						var ones = $M.CL.mapGenerator('1.0');
						return function(xs) {
							ones(xs);
							return xs;
						};
					}();
				} else {
					this.activate_function = function(xs) {
						var sum = $M.sumEachCol(xs.map(Math.exp));
						xs.setEach(function(row, col) {
							return xs.get(row, col) / sum.get(0, col);
						});
						return xs;
					};
					this.activate_function_div = function(xs) {
						xs.zeros(1.0);
						return xs;
					}
				}
				break;
			case 'relu':
				if ($M.CL) {
					this.activate_function = function() {
						var relu = $M.CL.mapGenerator('(a[i] > 0) ? a[i] : 0');
						return function(xs) {
							relu(xs);
							return xs;
						};
					}();
					this.activate_function_div = function() {
						var relu_div = $M.CL.mapGenerator('(a[i] > 0) ? 1 : 0');
						return function(xs) {
							relu_div(xs);
							return xs;
						};
					}();
				} else {
					this.activate_function = function(xs) { return xs.map(function(x) { return (x > 0) ? x : 0 }); };
					this.activate_function_div = function(xs) { return xs.map(function(x) { return (x > 0 ? 1 : (x === 0 ? 0 : -1)); }); };
				}
				break;
			case 'tanh':
				if ($M.CL) {
					this.activate_function = function() {
						var tanh = $M.CL.mapGenerator('(tanh(a[i]) + 1.0) / 2.0');
						return function(xs) {
							tanh(xs);
							return xs;
						};
					}();
					this.activate_function_div = function() {
						var tanh_div = $M.CL.mapGenerator('(1 - a[i] * a[i]) / 2.0');
						return function(xs) {
							tanh_div(xs);
							return xs;
						};
					}();
				} else {
					this.activate_function = function(xs) { return xs.map(function(x) { return (Math.tanh(x) + 1.0) / 2.0 }); };
					this.activate_function_div = function(xs) { return xs.map(function(x) { return (1.0 - x * x) / 2.0; }); };
				}
				break;
			default:
				throw new Error('the activation type is not supported.');
				break;
		}
	};

	Sukiyaki.Layers.ActivateLayer.prototype.update = function() { };

	Sukiyaki.Layers.ActivateLayer.prototype.calculateUpdateParams = function() { };

	Sukiyaki.Layers.ActivateLayer.prototype.forward = function() {
		this.output = this.input.largeClone();
		this.activate_function(this.output);
	};

	Sukiyaki.Layers.ActivateLayer.prototype.backward = function() {
		// console.log('activate start backward');
		var deactivated = this.activate_function_div(this.output.largeClone());
		this.delta_input = $M.largeMulEach(this.delta_output, deactivated);
		deactivated.destruct();
	};

	Sukiyaki.Layers.ActivateLayer.prototype.saveToJson = function() { return {}; };
	Sukiyaki.Layers.ActivateLayer.prototype.loadFromJson = function(data) {  };
})(typeof window === 'undefined', Sushi.Matrix);
/* end : layers/activate_layer.js */

/* begin : layers/convolutional_layer.js */
(function (nodejs, $M) {
	if (nodejs) {
		
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
/* end : layers/convolutional_layer.js */

/* begin : layers/fully_connected_layer.js */
(function (nodejs, $M) {
	if (nodejs) {
		
	}
	
	Sukiyaki.Layers.FullyConnectedLayer = function(params) {
		// variables
		this.w = null;
		this.b = null;
		
		// constructor
		this.checkContainParams(params, ['input_size', 'output_size']);
		this.initParams(params.input_size, params.output_size);
	};
	Sukiyaki.Layers.FullyConnectedLayer.prototype = new Sukiyaki.Layers.LayerBase();
	
	Sukiyaki.Layers.FullyConnectedLayer.prototype.initParams = function(input_size, output_size) {
		// w and b
		this.w = new $M(output_size, input_size);
		this.w.gaussRandom(0, Math.sqrt(1.0 / input_size));
		this.b = new $M(output_size, 1);
		this.b.largeZeros();
		// for AdaGrad
		this.adagrad_w = new $M(output_size, input_size);
		this.adagrad_w.largeZeros(0.01);
		this.adagrad_b = new $M(output_size, 1);
		this.adagrad_b.largeZeros(0.01);
		// utility parameters
		this.input_size = input_size;
		this.output_size = output_size;
	};
	
	Sukiyaki.Layers.FullyConnectedLayer.prototype.update = function() {
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
	
	Sukiyaki.Layers.FullyConnectedLayer.prototype.calculateUpdateParams = function() {
		this.delta_w = this.delta_output.largeMul(this.input.t()).largeTimes(1.0 / this.input.cols);
		this.delta_b = $M.largeSumEachRow(this.delta_output).largeTimes(1.0 / this.input.cols);
	};
	
	Sukiyaki.Layers.FullyConnectedLayer.prototype.calcAdaGrad = function() {
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
	
	Sukiyaki.Layers.FullyConnectedLayer.prototype.forward = function() {
		if (this.input.rows !== this.input_size) {
			throw new Error('input size does not match');
		}
		this.output = $M.largeMul(this.w, this.input);
		this.output.largeAdd(this.b);
	};
	
	Sukiyaki.Layers.FullyConnectedLayer.prototype.backward = function() {	
		if (this.delta_output.rows !== this.output_size) {
			throw new Error('output size does not match');
		}
		this.delta_input = $M.largeMul(this.w.t(), this.delta_output);
	};
	
	Sukiyaki.Layers.FullyConnectedLayer.prototype.destruct = function() {
		this.w.destruct();
		this.b.destruct();
	};
	
	Sukiyaki.Layers.FullyConnectedLayer.prototype.saveToJson = function() {
		this.w.syncData();
		this.b.syncData();
		return {
			w : this.w.data.buffer.toBase64(),
			b : this.b.data.buffer.toBase64()
		};
	};
	Sukiyaki.Layers.FullyConnectedLayer.prototype.loadFromJson = function(data) {
		this.w.syncData();
		this.b.syncData();
		this.w.data.buffer.fromBase64(data.w);
		this.b.data.buffer.fromBase64(data.b);
	};
})(typeof window === 'undefined', Sushi.Matrix);
/* end : layers/fully_connected_layer.js */

/* begin : layers/max_pooling_layer.js */
(function (nodejs, $M) {
	if (nodejs) {
		
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
/* end : layers/max_pooling_layer.js */
