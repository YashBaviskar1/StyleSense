function upload() {
  const fileUploader = document.querySelector('#image-upload');
  const image = fileUploader.files[0];
  const fileReader = new FileReader();
  fileReader.readAsDataURL(image);

  fileReader.onload = (fileReaderEvent) => {
      const profilePicture = document.querySelector('.image-preview');
      profilePicture.style.backgroundImage = `url(${fileReaderEvent.target.result})`;
  }
}

// AJAX to override the default event listener --> Solving ths issue of : reloading the page and image preview getting lost 
document.querySelector('form').addEventListener('submit', function(event) {
  event.preventDefault();  
  const formData = new FormData(this);
  fetch('/recommendation', {
      method: 'POST',
      body: formData
  })
  .then(response => response.text())
  .then(html => {
      console.log("Form submitted successfully.");
  })
  .catch(error => console.error('Error:', error));
});
