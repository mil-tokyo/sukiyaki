(function (nodejs, $M) {
	if (nodejs) {
		require('./layer_base');
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