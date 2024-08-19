
function upload(){
    const fileUploader = document.querySelector('#image-upload')
    const image = fileUploader.files[0]
    const fileReader = new FileReader();
    fileReader.readAsDataURL(image);
  
    fileReader.onload = (fileReaderEvent) => {
      const profilePicture = document.querySelector('.image-preview');
      profilePicture.style.backgroundImage = `url(${fileReaderEvent.target.result})`;
    }
}