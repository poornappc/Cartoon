# Cartoonizer: Combining CartoonGAN and White-Box Cartoon Representations

## Description
Cartoonizer is an interactive web application that transforms regular images into cartoon-like visuals. Utilizing state-of-the-art models like CartoonGAN and White-box Cartoonization, the application ensures high-quality cartoon effects that are visually appealing and closely resemble hand-drawn art.

![Image Result](https://github.com/mayank-vekariya/cartoon_web_app/assets/75078887/0168b208-06ec-49e4-aac0-49e0417c800d)
![Video Result](https://github.com/mayank-vekariya/cartoon_web_app/assets/75078887/1657b25b-5024-4e1b-b166-99198b5539db)

[Demo Video](https://www.youtube.com/watch?v=tiok_jClQ0k)

## Project Structure

```plaintext
cartoon/
│
├── app.py                              # Main application file running the web server and handling requests
├── final.yaml                          # YAML configuration file
│
├── cartoonGAN/                         # Contains implementation and assets related to the CartoonGAN model
│   ├── generator_latest.pth           # Pre-trained generator model
│   └── transform.py                   # Transformation script using CartoonGAN
│
├── white_box_cartoonization/          # Contains implementation and assets for the White-box Cartoonization model
│   ├── cartoonize.py                  # Script to cartoonize images using White-box Cartoonization
│   └── saved_models/                  # Directory containing model files for white box cartoonization
│
├── static/                            # Directory to store static assets like images
│   ├── uploads/                       # Sample uploaded files
│   └── cartoons/                      # Sample cartoon images
│
└── templates/                         # Contains HTML templates defining the structure and look of web pages
    ├── index.html                     # Main HTML template for the web application
```

- **app.py**: Main application file that runs the web server and handles user requests.
- **cartoonGAN**: Implementation and assets related to the CartoonGAN model.
- **white_box_cartoonization**: Implementation and assets for the White-box Cartoonization model.
- **static**: Stores static assets like images.
- **templates**: Contains HTML templates which define the structure and look of the web pages.
- **final.yaml**: A YAML configuration file, possibly related to the environment or other settings.

## Setting Up the Project

### Using Anaconda (Recommended)
1. Ensure that you have Anaconda or Miniconda installed.
2. Navigate to the project directory.
3. Create a new conda environment using the provided final.yaml file:conda env create -f final.yaml
4. Activate the newly created environment:conda activate <env_name>

Replace `<env_name>` with the environment name specified in the final.yaml file.

## Running the Web Application
1. Ensure you're in the project directory.
2. Run the app.py script:python app.py
3. Open a web browser and navigate to the address shown in the console (usually http://127.0.0.1:5000/).

## Behind the Scenes
- **CartoonGAN**: Employs Generative Adversarial Networks, trained on a vast dataset of real-world images and their cartoon versions.
- **White-box Cartoonization**: A model providing a unique cartoon style, allowing users to choose between different cartoonization effects.


## Important Links
- [Data/Files](https://drive.google.com/drive/folders/1tdEgYpGkWubJiqRCOqF-y9FbZOSSFel_?usp=sharing)
- [Kaggle Dataset](https://www.kaggle.com/alamson/safebooru)
- [COCO Dataset](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [GitHub Repo 1](https://github.com/bapu-1777/Real-to-Cartoon-Transformation-using-Generative-Adversarial-Networks)
- [GitHub Repo 2](https://github.com/bapu-1777/cartoon_web_app)
- [GitHub Repo 3](https://github.com/bapu-1777/https-github.com-bapu-1777-Real-to-Cartoon-Transformation-using-White-Box-Cartoon)

## Authors
- **Nitya Parikh** - UTA ID: 1002053645, NET ID: nxp3645
- **Mayank Vekariya** - UTA ID: 1002078999, NET ID: mxv8999

## Acknowledgments
- [Research Paper 1](https://www.irjet.net/archives/V7/i1/IRJET-V7I1376.pdf)
- [Research Paper 2](https://www.irjet.net/archives/V8/i4/PIT/ICIETET-45.pdf)
- [Research Paper 3](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)
- [More Research Papers and Resources](AdditionalLink)


