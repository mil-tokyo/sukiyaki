(function (nodejs, $M) {
	if (nodejs) {
		require('./layer_base');
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