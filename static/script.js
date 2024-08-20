function handleFileSelection() {
    var input = document.getElementById('file');
    var fileNamesDisplay = document.getElementById('file-names');
    var uploadBtn = document.getElementById('upload-btn');

    console.log("ERROR");

    if (input.files.length > 0) {
        var fileNames = Array.from(input.files).map(file => file.name).join(', ');
        fileNamesDisplay.innerText = 'Selected files : ' + fileNames;
        fileNamesDisplay.style.marginTop = '10px';
        fileNamesDisplay.style.fontSize = '16px';
        uploadBtn.disabled = false;
        uploadBtn.style.backgroundColor = '#1f255b'; // Même couleur que l'état activé
        uploadBtn.style.cursor = 'pointer';
    } else {
        fileNamesDisplay.innerText = '';
        uploadBtn.disabled = true;
        uploadBtn.style.backgroundColor = 'rgba(27,31,59,0.46)'; // Grise le bouton
        uploadBtn.style.cursor = 'not-allowed';
    }
}