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

// Prevent the form from refreshing the page when submitting
document.querySelector('form').addEventListener('submit', function(event) {
  event.preventDefault();  // Prevent the default form submission (page reload)
  
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
