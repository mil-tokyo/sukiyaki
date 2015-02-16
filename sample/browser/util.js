var Util = {
	consoleLog : function(txt) {
		txt = (new Date()).toLocaleString() + ' : ' + txt;
		var new_element = $("<div>").html(txt);
		$("#console").append(new_element);
		$("#console").scrollTop($("#console")[0].scrollHeight);
	}
};