'use strict';

require('../../sushi/src/sushi');
require('../../sushi/src/sushi_cl');
var $M = Sushi.Matrix;

var assert = require('assert');
var fs = require('fs');

var CifarLoader = function() {
};

CifarLoader.load = function(dir, files) {
	var rows = 32;
	var cols = 32;
	var depths = 3;
	
	var images = [];
	var labels = [];
	for (var i = 0; i < files.length; i++) {
		var bin = fs.readFileSync(dir + files[i]);
		for (var j = 0; j < 10000; j++) {
			var image = new $M(rows * depths, cols);
			image.syncData();
			for (var offset = 0; offset < rows * cols * depths; offset++) {
				image.data[offset] = bin.readUInt8(j * (rows * cols * depths + 1) + offset + 1) / 255.0;
			}
			images.push(image);
			labels.push(bin.readUInt8(j * (rows * cols * depths + 1)));
		}
	}
	return {
		images : images,
		labels : labels
	};
};

module.exports = CifarLoader;