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
		var jsdeep = new JSDeep(data.layer_params);
		for (var i = 0; i < jsdeep.layers.length; i++) {
			jsdeep.layers[i].loadFromJson(data.layer_status[i]);
		}
		return jsdeep;
	};
	
	(('global', eval)('this')).Sukiyaki = Sukiyaki;
})(typeof window === 'undefined', Sushi.Matrix);