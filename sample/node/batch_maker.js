'use strict';

require('../../sushi/src/sushi');
require('../../sushi/src/sushi_cl');
var $M = Sushi.Matrix;

var BatchMaker = function() {
}

BatchMaker.parseRawData = function(raw_data, batch_size) {
	var batches = [];
	var input_vectors = [];
	var output_vectors = [];
	var labels = [];
	for (var i = 0; i < raw_data.images.length; i++) {
		if (false && i === 3000) {
			break;
		}
		
		input_vectors.push(raw_data.images[i].reshape(raw_data.images[i].length, 1));
		output_vectors.push(new $M(10, 1).set(raw_data.labels[i], 0, 1));
		labels.push(raw_data.labels[i]);
		
		if (input_vectors.length === batch_size) {
			batches.push({
				input : $M.hstack(input_vectors),
				output : $M.hstack(output_vectors),
				labels : labels
			});
			input_vectors = [];
			output_vectors = [];
			labels = [];
		}
	}
	if (input_vectors.length > 0) {
		batches.push({
			input : $M.fromColVectors(input_vectors),
			output : $M.fromColVectors(output_vectors),
			labels : labels
		});
	}
	return batches;
};

module.exports = BatchMaker;