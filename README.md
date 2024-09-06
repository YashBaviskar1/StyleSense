# StyleSense
StyleSense is a unique fashion inventory app which combines power of CNN and content-based filtering to help users to classify various clothing items as well as find similar type of products based on the user uploads, as well as curating a pre-defined styles section to help users to enhance their fashion taste.  


# Setup and Installation 
To run this project on your location machine follow these steps

- clone this project 
```bash 
git clone https://github.com/YashBaviskar1/StyleSense.git
```
- once installed go in the clone directory 
```bash
cd StyleSense
```
- make sure virtualvenv is installed, to install venv to host python run this command 
``` bash
pip install virtualenv
```
- open the terminal in the StyleSense directory and run this command 
```bash
 python -m venv myenv
```
- to activate the virtual enviroment in your terminal run these commands 
```bash
 myenv/Scripts/activate.bat //In CMD
 myenv/Scripts/Activate.ps1 //In Powershell
```
- to check if virtual enviroment is active, you will see a (myenv) in your terminal, it means you have successfuly activated the virtual enviroment 

- to check the dependencies you can run this command 
```bash
pip list
```
- to install these dependecies you can run the following command (this will take time)
```bash
pip install -r requirements.txt
```

- in order. to run the project you can flask run in your terminal
```bash
flask run
```
- This will start a local server. You can view the web application in your browser.and to To quit the server, press Ctrl + C.
