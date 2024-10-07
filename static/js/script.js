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

// AJAX to override the default event listener >------ Solving ths issue of : reloading the page and image preview getting lost. ----NOTE--->[this is old now, keeping it here as a relic and to remind myself of my failures in life , i wasted so long on this just to remove it completely ]
// >------------- ({new edit, this ruined my life fuck you }) -------------------<
// document.querySelector('form').addEventListener('submit', function(event) {
//   event.preventDefault();  
//   const formData = new FormData(this);
//   fetch('/recommendation', {
//       method: 'POST',
//       body: formData
//   })
//   .then(response => response.text())
//   .then(html => {
//       console.log("Form submitted successfully.");
//   })
//   .catch(error => console.error('Error:', error));
// });

// async function getResponse() {
//   try {
//     const response = await fetch("http://127.0.0.1:5000/recommendation", {
//       method: 'GET',
//       headers: {
//         'Content-Type': 'application/json'
//       }
//     });

//     // Await JSON data from the response
//     const data = await response.json();
//     console.log(data.recommended_images);

//     // Dynamically display images on the page
//     displayImages(data.recommended_images);

//   } catch (error) {
//     console.error('Error fetching recommendation:', error);
//   }
// }

// Call the function to execute it
// GetResponse();

// ----> this is your classic JSON from the app.py, ENJOY, screw you ajax you will not be missed <----
async function handleFormSubmit(event) {
  event.preventDefault(); 

  const form = document.getElementById('upload-form');
  const formData = new FormData(form); 

  // POST request to the server with the image data
  const response = await fetch('/recommendation', {
      method: 'POST',
      body: formData
  });

  // >------ Get the JSON response after the server processes it  ------<
  const data = await response.json();
  
  console.log(data.recommended_images)
  displayImages(data.recommended_images);
}
// ##### >-------------- Display Images handling, yes i know it sucks i will make it scalable later -------------< #####
function displayImages(images){
  for (let i = 0; i < images.length; i++) {
    const imagePath = images[i].replace(/\\/g, '/');
    

    document.getElementById(`rec-image${i + 1}`).style.backgroundImage = `url(${imagePath})`;
    const saveButton = document.getElementById(`save-button${i + 1}`);
    saveButton.style.display = 'inline-block';  
    saveButton.onclick = () => saveImage(imagePath);
  }
}


function saveImage(imagePath) {
    fetch('/save_recommendation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image_path: imagePath })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Image saved successfully');
        } else {
            alert('Error saving image');
        }
    });
}



// ##-CLASSIFICATION-OUTFIT-GENERATION-MODULE 
