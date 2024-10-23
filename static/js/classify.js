function upload2() {
    const fileUploader = document.querySelector('#image-upload');
    const image = fileUploader.files[0];
    const fileReader = new FileReader();
    fileReader.readAsDataURL(image);
    console.log(image)
    fileReader.onload = (fileReaderEvent) => {
        const profilePicture = document.querySelector('.image-preview');
        profilePicture.style.backgroundImage = `url(${fileReaderEvent.target.result})`;
        console.log(fileReaderEvent.target.result)
        document.querySelector('#predict-button').click();
    }
  }
async function classify(event){
    event.preventDefault();
    const formData = new FormData();
    const fileUploader = document.querySelector('#image-upload');
    const image = fileUploader.files[0];
    formData.append('image', image);
  
    try {
        const response = await fetch('/classify', {
            method: 'POST',
            body: formData,
        });
  
        const result = await response.json();
        const predictedLabel = result.category;
        console.log(predictedLabel)
    
        document.querySelector('.classify').innerHTML = `Category: ${predictedLabel}`;
    } catch (error) {
        console.error('Error:', error);
    }
  }


  