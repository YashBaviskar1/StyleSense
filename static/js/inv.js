function inventoryUploads() {
    const input = document.getElementById('upload');
    const previewContainer = document.getElementById('image-preview');
    const autoClassifyBtn = document.getElementById('autoclassify-btn');
    const submitInventoryBtn = document.getElementById('submit-btn');
    previewContainer.innerHTML = ''; 

    if (input.files && input.files.length > 0) {
        autoClassifyBtn.style.display = 'block';
        autoClassifyBtn.classList.add('button')
        console.log(input.files)

        Array.from(input.files).forEach((file, index) => {
            const reader = new FileReader();

            reader.onload = function(e) {
                const imgContainer = document.createElement('div');
                imgContainer.classList.add('img-container');              
                const img = document.createElement('div');
                img.style.backgroundImage = `url(${e.target.result})`;
                // console.log(e.target.result)
                img.classList.add('image-preview');             
                const label = document.createElement('p');
                label.textContent = 'Category: ';
                label.classList.add('category-label', `category-label${index + 1}`);
                const editBtn = document.createElement('button');
                editBtn.textContent = 'Edit';
                editBtn.classList.add('edit-btn', `edit-btn${index + 1}`);

                editBtn.addEventListener('click', () => {
                    const userInput = prompt("Enter your Category:");
                    if (userInput) {
                        label.textContent = `Category: ${userInput}`;
                    }
                });


                imgContainer.appendChild(img);
                imgContainer.appendChild(label);
                imgContainer.appendChild(editBtn);
                
                previewContainer.appendChild(imgContainer);
            };

            reader.readAsDataURL(file);
        });
        submitInventoryBtn.style.display = 'block';
        submitInventoryBtn.classList.add('button')


    } else {
        autoClassifyBtn.style.display = 'none'
        autoClassifyBtn.style.margin = '10px';
    }
}

function autoClassify() {
    const input = document.getElementById('upload');
    const formData = new FormData();
    
    for (let i = 0; i < input.files.length; i++) {
        formData.append('upload', input.files[i]);
    }

    fetch('/inventory', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const categories = data.categories;
        console.log(categories)
        categories.forEach((category, index) => {
            const label_classify = document.getElementsByClassName(`category-label${index + 1}`)[0];
            label_classify.textContent = `Category: ${category}`;
        });
    })
    .catch(error => {
        console.error('Error during auto-classification:', error);
    });
}


function submitInventory(){
    const categories = [];
    document.querySelectorAll('.category-label').forEach((label) => {
        categories.push(label.textContent.replace('Category: ', '').trim());
    });
    fetch('/inventory/save', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ categories: categories }),  
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        alert('Categories have been saved successfully!');
    })
    console.log('Categories saved:', categories);
    alert('Categories have been saved successfully!');
}