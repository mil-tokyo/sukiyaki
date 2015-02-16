var LoaderForBrowser = function() {
	this.notifier = function(msg) { console.log(msg); };
};

LoaderForBrowser.prototype.setNotifier = function(notifier) {
	this.notifier = notifier;
};

LoaderForBrowser.prototype.getFilesOrData = function(url, callback) {
	var xhr = new XMLHttpRequest();
	xhr.open('GET', url, true);
	xhr.responseType = 'arraybuffer';
	xhr.onload = function(e) {
		callback(xhr.response);
	};
	xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
	xhr.send();
};