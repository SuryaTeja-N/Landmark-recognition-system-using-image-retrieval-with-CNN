function uploadImage() {
    const input = document.getElementById('imageInput');
    const file = input.files[0];

    if (!file) {
        alert("Please select an image first.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    console.log("Uploading image...");

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log("Response received:", response);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        document.getElementById('landmarkName').innerText = `Landmark: ${data.landmark_name}`;
        const similarImagesDiv = document.getElementById('similarImages');
        similarImagesDiv.innerHTML = "";

        if (data.similar_images.length === 0) {
            similarImagesDiv.innerText = "No similar images found.";
            return;
        }

        data.similar_images.forEach((imageSrc, index) => {
            const img = document.createElement('img');
            img.src = imageSrc;
            img.alt = `Similar Image ${index + 1}`;
            img.style.width = '200px';
            img.style.marginRight = '10px';
            similarImagesDiv.appendChild(img);
        });
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while uploading the image.');
    });
}