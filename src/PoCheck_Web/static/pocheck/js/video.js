(function() {
    var canvas = document.querySelector('canvas'),
        context = canvas.getContext('2d'),
        video = document.querySelector('video'),
        vendorUrl = window.URL || window.webkitURL;

    navigator.getUserMedia = navigator.getUserMedia ||
                         navigator.webkitGetUserMedia ||
                         navigator.mozGetUserMedia ||
                         navigator.msGetUserMedia;
    navigator.getUserMedia({
        video: true,
        audio: false
    }, function(stream) {
        video.srcObject = stream;
        video.play();
    }, function(error) {
        // An error occured
        console.log('navigator.getUserMedia error: ', error);
    });

    video.addEventListener('play', function() {
        draw(this, context, 400, 300 )
    }, false);
    function draw(video, context, width, height) {
        context.drawImage(video, 0, 0, width, height);
        setTimeout(draw, 10, video, context, width, height);
    }
})();