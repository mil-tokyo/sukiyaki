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