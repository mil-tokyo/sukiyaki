'use strict';

require('../../sushi/src/sushi');
require('../../sushi/src/sushi_cl');
var $M = Sushi.Matrix;

require('../../bin/sukiyaki');
var MNISTLoader = require('./mnist_loader');
var CifarLoader = require('./cifar_loader');
var BatchMaker = require('./batch_maker');

if ($M.CL) {
	console.log('using device : ' + $M.CL.device_info + ' (' + $M.CL.platform_info + ')');
}

var SampleMain = function(dataset) {
	this.dataset = dataset;
	this.batch_size = 50;
	this.epoch = 0;
};

SampleMain.prototype.start = function() {
	console.log('using dataset : ' + this.dataset);
	
	console.log('start loading data');
	this.loadData();
	console.log('done');
	
	console.log('start initializing network');
	this.initNetwork();
	console.log('done');
	
	this.setNotifier();
	
	console.log('start training with ' + this.train_raw_data.images.length + ' images in ' + this.train_batches.length + ' batches');
	this.learnLoop();
};

SampleMain.prototype.loadData = function() {
	switch (this.dataset) {
		case 'mnist':
			this.train_raw_data = MNISTLoader.load(
				absolute_path('../dataset/mnist/train-images-idx3-ubyte'),
				absolute_path('../dataset/mnist/train-labels-idx1-ubyte')
			);
			this.test_raw_data = MNISTLoader.load(
				absolute_path('../dataset/mnist/t10k-images-idx3-ubyte'),
				absolute_path('../dataset/mnist/t10k-labels-idx1-ubyte')
			);
			break;
		case 'cifar':
			this.train_raw_data = CifarLoader.load(absolute_path('../dataset/cifar/'), ['data_batch_1.bin', 'data_batch_2.bin', 'data_batch_3.bin', 'data_batch_4.bin', 'data_batch_5.bin']);
			this.test_raw_data = CifarLoader.load(absolute_path('../dataset/cifar/'), ['test_batch.bin'])
			break;
	}
	this.train_batches = BatchMaker.parseRawData(this.train_raw_data, this.batch_size);
	this.test_batches = BatchMaker.parseRawData(this.test_raw_data, this.batch_size);
	
	function absolute_path(relative) {
		var path = require('path');
		return path.resolve(__dirname, relative);
	}
};

SampleMain.prototype.initNetwork = function() {
	switch (this.dataset) {
		case 'mnist':
			this.sukiyaki = new Sukiyaki([
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
             ]);
			break;
		case 'cifar':
			this.sukiyaki = new Sukiyaki([
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
			]);
			break;
	}
};

SampleMain.prototype.setNotifier = function() {
	this.sukiyaki.setNotifier(function(data) {
		switch(data.type) {
			case 'learn_batches_finished_count':
				console.log(data.params.batch_count + ' / ' + data.params.batch_all + ' batches finished.');
				break;
		}
	});
};

SampleMain.prototype.predict = function() {
	var results = [];
	var confusion_mat = new $M(10, 10);
	confusion_mat.syncData();
	var i = 0;
	var correct = 0;
	return function() {
		for (var batches_to_predict_at_once = 0; batches_to_predict_at_once < 100; batches_to_predict_at_once++) {
			// predict
			var predicts = this.sukiyaki.predict(this.test_batches[i].input);
			predicts.syncData();
			var labels = this.test_batches[i].labels;
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
			i++;
			i %= this.test_batches.length;
			predicts.destruct();
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
		console.log('error_rate : ' + String((1.0 - correct / results.length) * 100.0) + ' %');
		confusion_mat.print();
	};
}();

SampleMain.prototype.learnLoop = function() {
	var learn_loop_start_time = (new Date()).getTime();
	console.log('epoch #' + this.epoch)
	var from = 0;
	var batch_per_once = 100;
	var batch_loop = function() {
		var last_time = (new Date()).getTime();
		var to = Math.min(this.train_batches.length, from + batch_per_once);
		var last = (to === this.train_batches.length);
		var fitting_error = this.sukiyaki.learn(this.train_batches, from, to, true);
		var elapsed_time = (new Date()).getTime() - last_time;
		console.log('time for each batch : ' + (elapsed_time / (to - from)) + ' ms');
		if (fitting_error !== void 0) {
			if (!fitting_error) {
				console.error(fitting_error);
			}
			console.log('fitting error : ' + fitting_error);
		}
		if (to % 100 === 0) {
			this.predict();
		}
		if (last) {
			this.epoch++;
			if (this.epoch < 30) {
				console.log('elapsed_time in this epoch : ' + ((new Date()).getTime() - learn_loop_start_time) + ' ms');
				setTimeout(this.learnLoop.bind(this), 1);
			}
		} else {
			from += batch_per_once;
			setTimeout(batch_loop.bind(this), 1);
		}
	};
	batch_loop.bind(this)();
};

(function() {
	/*
	console.log('Choose dataset (cifar or mnist)');
	var readline = require("readline").createInterface(process.stdin, process.stdout);
	readline.question("> ", function(value){
		console.log('');
		value = value.trim();
		if (value !== 'cifar' && value !== 'mnist') {
			console.log('Invalid input');
		} else {
			var sample_main = new SampleMain(value);
			sample_main.start();
		}
	    readline.close();
	});*/
	var sample_main = new SampleMain('mnist');
	sample_main.start();
})();
