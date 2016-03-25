'use strict';

var $M = Sushi.Matrix;

var MnistLoaderForBrowser = function(dir, files, max, batch_size, callback) {
	this.dir = dir;
	this.files = files;
	this.max = max;
	this.batch_size = batch_size;
	this.all_loaded = 0;
	this.batches = [];
	this.images_dataview = null;
	this.labels_dataview = null;
	this.bin_load_index = 0;
	this.callback = callback;
	if (files.length !== 2) {
		throw new Error('file names must be [image, label]');
	}
};

MnistLoaderForBrowser.prototype = new LoaderForBrowser();

MnistLoaderForBrowser.prototype.loadFile = function() {
	this.getFilesOrData(this.dir + this.files[0] + '.deflate', function(ab) {
    var uncompressed = zlib.inflate(new Uint8Array(ab));
		this.images_dataview = new DataView(uncompressed.buffer);
		this.startParseBatch();
	}.bind(this));
	this.getFilesOrData(this.dir + this.files[1] + '.deflate', function(ab) {
    var uncompressed = zlib.inflate(new Uint8Array(ab));
		this.labels_dataview = new DataView(uncompressed.buffer);
		this.startParseBatch();
	}.bind(this));
};

MnistLoaderForBrowser.prototype.startParseBatch = function() {
	if (!this.images_dataview || !this.labels_dataview) {
		return;
	}					
	if (this.images_dataview.getInt32(0) !== 2051) {
		throw new Error('invalid datafile');
	}
	if (this.labels_dataview.getInt32(0) !== 2049) {
		throw new Error('invalid datafile');
	}
	var images_num = this.images_dataview.getInt32(4);
	if (images_num !== this.labels_dataview.getInt32(4)) {
		throw new Error("number of images and labels doesn't match");
	}
	if (this.max) {
		this.max = Math.min(this.max, images_num);
	} else {
		this.max = images_num;
	}
	this.parseNextBatch();
};

MnistLoaderForBrowser.prototype.parseNextBatch = function() {
	if (this.bin_load_index === this.max) {
		this.callback(this.batches, this.all_loaded);
		return;
	}
	var rows = 28;
	var cols = 28;
	var batch_size = Math.min(this.batch_size, (this.max - this.bin_load_index));
	var input = new $M(rows * cols, batch_size);
	var output = new $M(10, batch_size);
	input.syncData();
	output.syncData();
	var labels = [];
	for (var batch = 0; batch < batch_size; batch++) {
		var load_offset = 16 + this.bin_load_index * (rows * cols);
		var save_offset = batch;
		for (var row = 0; row < rows; row++) {
			for (var col = 0; col < cols; col++) {
				input.data[save_offset] = this.images_dataview.getUint8(load_offset) / 255.0;
				load_offset++;
				save_offset += batch_size;
			}
		}
		var label = this.labels_dataview.getUint8(8 + this.bin_load_index);
		output.data[batch_size * label + batch] = 1;
		labels.push(label);
		this.bin_load_index++;
		this.all_loaded += 1;
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

MnistLoaderForBrowser.load = function(dir, files, max, batch_size, notifier, callback) {
	var mnist_loader_for_browser = new MnistLoaderForBrowser(dir, files, max, batch_size, callback);
	if (notifier) {
		mnist_loader_for_browser.notifier = notifier;
	}
	mnist_loader_for_browser.loadFile();
};
