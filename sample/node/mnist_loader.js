'use strict';

require('milsushi');
var $M = Sushi.Matrix;

var assert = require('assert');
var fs = require('fs');

var MNISTLoader = function() {
};

MNISTLoader.load = function(images_path, labels_path) {
	var images_bin = fs.readFileSync(images_path);
	var labels_bin = fs.readFileSync(labels_path);
	
	assert.strictEqual(images_bin.readInt32BE(0), 2051);
	assert.strictEqual(labels_bin.readInt32BE(0), 2049);
	var images_num = images_bin.readInt32BE(4);
	assert.strictEqual(images_num, labels_bin.readInt32BE(4));
	var rows = images_bin.readInt32BE(8);
	var cols = images_bin.readInt32BE(12);
	
	var images = [];
	var labels = [];
	for (var i = 0; i < images_num; i++) {
		var offset = 16 + i * (rows * cols);
		var image = new $M(rows, cols);
		for (var y = 0; y < rows; y++) {
			for (var x = 0; x < cols; x++) {
				image.set(y, x, images_bin.readUInt8(offset + x + y * cols) / 255.0);
			}
		}
		images.push(image);
		labels.push(labels_bin.readUInt8(8 + i));
	}
	return {
		images : images,
		labels : labels
	};
};

module.exports = MNISTLoader;
