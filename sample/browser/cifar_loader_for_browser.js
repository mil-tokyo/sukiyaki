'use strict';

var $M = Sushi.Matrix;

var CifarLoaderForBrowser = function(dir, files, max, batch_size, callback) {
	this.dir = dir;
	this.files = files;
	if (max) {
		max = Math.min(10000, max);
	} else {
		max = 10000;
	}
	this.max = max;
	this.batch_size = batch_size;
	this.all_loaded = 0;
	this.batches = [];
	this.dataview = null;
	this.bin_load_index = 0;
	this.callback = callback;
};

CifarLoaderForBrowser.prototype = new LoaderForBrowser();

CifarLoaderForBrowser.prototype.loadFile = function() {
	if (this.files.length === 0) {
		this.callback(this.batches, this.all_loaded);
	} else {
		var file = this.files.shift();
		this.getFilesOrData(this.dir + file, function(ab) {
			this.dataview = new DataView(ab);
			this.bin_load_index = 0;
			this.parseNextBatch();
		}.bind(this));
	}
};

CifarLoaderForBrowser.prototype.parseNextBatch = function() {
	if (this.bin_load_index === this.max) {
		this.loadFile();
		return;
	}
	var rows = 32;
	var cols = 32;
	var depths = 3;
	var batch_size = Math.min(this.batch_size, (this.max - this.bin_load_index));
	var input = new $M(rows * cols * depths, batch_size);
	var output = new $M(10, batch_size);
	input.syncData();
	output.syncData();
	var labels = [];
	for (var batch = 0; batch < batch_size; batch++) {
		var load_offset = this.bin_load_index * (rows * cols * depths + 1) + 1;
		var save_offset = batch;
		for (var depth = 0; depth < depths; depth++) {
			for (var row = 0; row < rows; row++) {
				for (var col = 0; col < cols; col++) {
					input.data[save_offset] = this.dataview.getUint8(load_offset) / 255.0;
					load_offset++;
					save_offset += batch_size;
				}
			}
		}
		var label = this.dataview.getUint8(this.bin_load_index * (rows * cols * depths + 1));
		output.data[batch_size * label + batch] = 1;
		labels.push(label);
		this.bin_load_index++;
		this.all_loaded++;
		if (this.all_loaded % 100 === 0) {
			this.notifier(this.all_loaded + ' images were loaded');
		}
	}
	this.batches.push({
		input : input,
		output : output,
		labels : labels
	});
	setTimeout(this.parseNextBatch.bind(this));
};

CifarLoaderForBrowser.load = function(dir, files, max, batch_size, notifier, callback) {
	var cifar_loader_for_browser = new CifarLoaderForBrowser(dir, files, max, batch_size, callback);
	if (notifier) {
		cifar_loader_for_browser.notifier = notifier;
	}
	cifar_loader_for_browser.loadFile();
};