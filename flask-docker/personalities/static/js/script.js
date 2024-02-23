document.addEventListener("DOMContentLoaded", function() {

    // Example JavaScript to interact with your form or elements
    const uploadForm = document.querySelector('#uploadForm');

    if(uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            // You can add any pre-upload checks or functionality here
            console.log('Video upload started.');
        });
    }

});


function updateFileName() {
    var fileInput = document.getElementById('video');
    var fileNameDisplay = document.getElementById('file-name');
    var fileName = fileInput.files[0] ? fileInput.files[0].name : "No file chosen";

    fileNameDisplay.textContent = fileName; // Update the text content of the span
}

