var BrowserSampleMain = function(dataset) {
	this.batch_size = 20;
	switch (dataset) {
		case 'cifar':
			this.loader = CifarLoaderForBrowser;
			this.data_base_url = '/sample/dataset/cifar/';
			this.train_files = ['data_batch_1.bin', 'data_batch_2.bin', 'data_batch_3.bin', 'data_batch_4.bin', 'data_batch_5.bin'];
			this.test_files = ['test_batch.bin'];
			this.layers = [
				{ type : 'conv', params : { input_rows : 32, input_cols : 32, input_depth :  3, output_depth : 16, window_size : 5, padding : 2 } },
				{ type : 'act', params : { activation_type : 'relu' } },
				{ type : 'pool', params : { input_rows : 32, input_cols : 32, input_depth : 16, window_size : 2, stride : 2 } },
				
				{ type : 'conv', params : { input_rows : 16, input_cols : 16, input_depth : 16, output_depth : 20, window_size : 5, padding : 2 } },
				{ type : 'act', params : { activation_type : 'relu' } },
				{ type : 'pool', params : { input_rows : 16, input_cols : 16, input_depth : 20, window_size : 2, stride : 2 } },
				
				{ type : 'conv', params : { input_rows : 8, input_cols : 8, input_depth : 20, output_depth : 20, window_size : 5, padding : 2 } },
				{ type : 'act', params : { activation_type : 'relu' } },
				{ type : 'pool', params : { input_rows : 8, input_cols : 8, input_depth : 20, window_size : 2, stride : 2 } },
				
				{ type : 'fc', params : { input_size : 4 * 4 * 20, output_size : 10 } },
				{ type : 'act', params : { activation_type : 'softmax' } }
				];
			this.class2name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
			this.drawImage = function(mat) {
				var canvas = document.getElementById('predict_image');
				canvas.width = 32;
				canvas.height = 32;
				var context = canvas.getContext('2d');
				var bmp = context.createImageData(32, 32);
				var setPixel = function(x, y, r, g, b, a) {
					var st = (x + 32 * y) * 4;
					var dt = bmp.data;
					dt[st++] = r;
					dt[st++] = g;
					dt[st++] = b;
					dt[st++] = a;
				};
				mat.syncData();
				for (var row = 0; row < 32; row++) {
					for (var col = 0; col < 32; col++) {
						setPixel(col, row, mat.data[((32*32) * 0 + 32 * row + col) * mat.cols] * 255, mat.data[((32*32) * 1 + 32 * row + col) * mat.cols] * 255, mat.data[((32*32) * 2 + 32 * row + col) * mat.cols] * 255, 255);
					}
				}
				context.putImageData(bmp, 0, 0);
			};
			break;
		case 'mnist':
			this.loader = MnistLoaderForBrowser;
			this.data_base_url = '/sample/dataset/mnist/';
			this.train_files = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte'];
			this.test_files = ['t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'];
			this.layers = [
             	{ type : 'conv', params : { input_rows : 28, input_cols : 28, input_depth :  1, output_depth : 10, window_size : 5, padding : 0 } },
             	{ type : 'act', params : { activation_type : 'relu' } },
             	{ type : 'pool', params : { input_rows : 24, input_cols : 24, input_depth : 10, window_size : 2, stride : 2 } },
             	
             	{ type : 'conv', params : { input_rows : 12, input_cols : 12, input_depth : 10, output_depth : 12, window_size : 3, padding : 0 } },
             	{ type : 'act', params : { activation_type : 'relu' } },
             	{ type : 'pool', params : { input_rows : 10, input_cols : 10, input_depth : 12, window_size : 2, stride : 2 } },
             	
             	{ type : 'fc', params : { input_size : 5 * 5 * 12, output_size : 128 } },
             	{ type : 'act', params : { activation_type : 'sigmoid' } },

             	{ type : 'fc', params : { input_size : 128, output_size : 10 } },
             	{ type : 'act', params : { activation_type : 'softmax' } }
             ];
			this.class2name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
			this.drawImage = function(mat) {
				var canvas = document.getElementById('predict_image');
				canvas.width = 28;
				canvas.height = 28;
				var context = canvas.getContext('2d');
				var bmp = context.createImageData(28, 28);
				var setPixel = function(x, y, r, g, b, a) {
					var st = (x + 28 * y) * 4;
					var dt = bmp.data;
					dt[st++] = r;
					dt[st++] = g;
					dt[st++] = b;
					dt[st++] = a;
				};
				mat.syncData();
				for (var row = 0; row < 28; row++) {
					for (var col = 0; col < 28; col++) {
						var tmp = mat.data[(28 * row + col) * mat.cols] * 255;
						setPixel(col, row, tmp, tmp, tmp, 255);
					}
				}
				context.putImageData(bmp, 0, 0);
			};
			break;
		default:
			throw new Error('invalid dataset');
			break;
	}
	Util.consoleLog('using dataset : ' + dataset);
};
BrowserSampleMain.prototype.start = function() {
	this.loader.load(this.data_base_url, this.train_files, null, this.batch_size, Util.consoleLog, function(train_batches, all_loaded) {
		this.loader.load(this.data_base_url, this.test_files, null, this.batch_size, Util.consoleLog, function(test_batches) {
			this.train_images_length = all_loaded;
			this.learnAndPredict(train_batches, test_batches);
		}.bind(this));
	}.bind(this));
};

BrowserSampleMain.prototype.learnAndPredict = function(train_batches, test_batches) {
	this.train_batches = train_batches;
	this.test_batches = test_batches;
	
	this.sukiyaki = new Sukiyaki(this.layers);
	this.sukiyaki.setNotifier(function(data) {
		switch(data.type) {
			case 'learn_batches_finished_count':
				Util.consoleLog(data.params.batch_count + ' / ' + data.params.batch_all + ' batches finished.');
				break;
		}
	});

	Util.consoleLog('start training with ' + this.train_images_length + ' images in ' + this.train_batches.length + ' batches');

	var epoch = 0;
	var start_time = (new Date()).getTime();
	var learn_loop = function() {
		var learn_loop_start_time = (new Date()).getTime();
		Util.consoleLog('epoch #' + epoch)
		var from = 0;
		var batch_per_once = 1;
		var batch_loop = function() {
			// learn
			var last_time = (new Date()).getTime();
			var to = Math.min(this.train_batches.length, from + batch_per_once);
			var last = (to == this.train_batches.length);
			var fitting_error = this.sukiyaki.learn(this.train_batches, from, to, true);
			var elapsed_time = (new Date()).getTime() - last_time;
			Util.consoleLog('time for each batch : ' + (elapsed_time / (to - from)) + ' ms');
			// visualize
			$("#epoch").html(epoch);
			$("#batch").html(from);
			if (fitting_error !== void 0) {
				if (!fitting_error) {
					console.error(fitting_error);
				}
				Util.consoleLog('fitting error : ' + fitting_error.toFixed(5));
			}
			// predict
			this.predict();
			if (last) {
				epoch++;
				if (epoch < 30) {
					Util.consoleLog('elapsed_time in this epoch : ' + ((new Date()).getTime() - learn_loop_start_time) + ' ms');
					setTimeout(learn_loop.bind(this), 1000);
				}
				console.log('epoch #' + epoch + '; whole elapsed_time : ' + ((new Date()).getTime() - start_time) + ' ms; error_rate : ' + $("#error_rate").html());
			} else {
				from += batch_per_once;
				setTimeout(batch_loop.bind(this), 1);
			}
		};
		batch_loop.bind(this)();
	};
	learn_loop.bind(this)();
};

BrowserSampleMain.prototype.predict = function() {
	var results = [];
	var confusion_mat = new $M(10, 10);
	confusion_mat.syncData();
	var i = 0;
	var correct = 0;
	return function() {
		// predict
		var predicts = this.sukiyaki.predict(this.test_batches[i].input);
		predicts.syncData();
		var labels = this.test_batches[i].labels;
		$("#correct_answer").html(this.id2name(labels[0]));
		// aggregate and update confusion matrix
		for (var j = 0; j < labels.length; j++) {
			var predict = predicts.data[j];
			var answer = labels[j];
			if (answer === predict) {
				correct++;
			}
			confusion_mat.data[answer * 10 + predict]++;
			results.push({ answer : answer, predict : predict });
		}
		// limit the result history to loop only once
		while (results.length >= 10000) {
			var result = results.shift();
			if (result.answer === result.predict) {
				correct--;
			}
			confusion_mat.data[result.answer * 10 + result.predict]--;
		}
		// visualize
		this.drawImage(this.test_batches[i].input);
		$("#prediction_answer").html(this.id2name(predicts.data[0]));
		$("#error_rate").html(((1.0 - correct / results.length) * 100.0).toFixed(4) + ' %');
		$("#confusion_matrix").html(confusion_mat.toString());
		
		// prepare for next loop
		predicts.destruct();
		i++;
		i %= this.test_batches.length;
	};
}();

BrowserSampleMain.prototype.id2name = function() {
	return function(id) {
		return this.class2name[id];
	};
}();
