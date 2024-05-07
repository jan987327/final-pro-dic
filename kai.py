import streamlit as st
from PIL import Image
import os

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Frame, Button, Label, Text, END
from PIL import Image, ImageTk
from tkinter import BOTH



import cv2
from tkinter import filedialog, Text, END, Tk
from PIL import Image, ImageTk
import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
import pressure
import zones
import segmentation
import joblib



def main():
    st.title("Dyslexia Prediction using Handwriting")
    st.write("This is a simple web app that works based on machine learning techniques. This application can predict the presence of dyslexia from the handwriting sample of a person.")
    with st.container():
        st.write("---")
        image = st.file_uploader("Upload the handwriting sample that you want to test", type=["png"])
        if image is not None:
            st.write("Please review the image selected")
            st.write(image.name)
            image_uploaded = Image.open(image)
            image_uploaded.save("temp.jpg")
            st.image(image_uploaded, width=224)

        if st.button("Predict", help="click after uploading the correct image"):
            try:
                feature_array = get_feature_array("temp.png")
                result = score(feature_array)
                if result[0] == 1:
                    st.write("From the tests on this handwriting sample there is very slim chance that this person is sufferning from dyslexia or dysgraphia")
                else:
                    st.write("From the tests on this handwriting sample there is very high chance that this person is sufferning from dyslexia or dysgraphia")
            except:
                st.write("Something went wrong at the server end please refresh the application and try again")


    # Load logo image
   # logo_image = Image.open("logo.png")
    #st.image(logo_image, width=400)

    # Display a text widget
    #st.markdown("Waiting for Results...")

    # Button for handwritting prediction
    if st.button("Select Handwriting Image"):
        handwritting_prediction()

def Train_writting(self, event=None):
        global T
        contents="Handwritting Feature extraction and Training"
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)  
        CNN_DATA1=[]
        S_Data=[]
        S_label=[]
        cnt=0
        cw_directory = os.getcwd()
        H_dataset = cw_directory+r'\Dataset2'
        for filename in os.listdir(H_dataset):
            sub_dir=(H_dataset+'/' +filename)
            for img_name in os.listdir(sub_dir):
                img_dir=str(sub_dir+ '/' +img_name)
                print(img_dir)
                feature_matrix1 = feature_extraction.Feature_extraction(img_dir)
                #print(len(feature_matrix1))
                S_Data.append(feature_matrix1)
                S_label.append(int(filename))
            cnt+=1
            print(cnt)

        ## MLP Training      
        model1 = MLPClassifier(activation='relu', verbose=True,
                                               hidden_layer_sizes=(100,), batch_size=30)
        model1=model1.fit(np.array(S_Data), np.array(S_label))
        ypred_MLP = model1.predict(np.array(S_Data))

        pair_confusion_matrix(model1, np.array(S_Data), np.array(S_label))
        plt.show()
        S_ACC=accuracy_score(S_label,ypred_MLP)

        print("Training ANN accuracy is",accuracy_score(S_label,ypred_MLP))
        joblib.dump(model1, "Trained_H_Model.pkl")


        ## Train SVM
        from sklearn.svm import SVC
        def train_SVM(featuremat,label):
            clf = SVC(kernel = 'rbf', random_state = 0)
            clf.fit(np.array(S_Data), np.array(S_label))
            y_pred = clf.predict(np.array(featuremat))
            pair_confusion_matrix(clf, np.array(featuremat), np.array(label))
            plt.show()
            print("SVM Accuracy",accuracy_score(label,y_pred))
            return clf

        svc_model1 = train_SVM(S_Data,S_label)
        Y_SCM_S_pred= svc_model1.predict(S_Data)
        SVM_S_ACC=accuracy_score(Y_SCM_S_pred,S_label)


        plt.figure()
        plt.bar(['ANN'],[S_ACC], label="ANN Accuracy", color='r')
        plt.bar(['SVM'],[SVM_S_ACC], label="SVM Accuracy", color='g')
        plt.legend()
        plt.ylabel('Accuracy')
        plt.show()
        contents="Handwritting Feature extraction and Training completed"
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)

def handwritting_prediction1():
    st.write("Handwritting Feature extraction and Training")
    
    CNN_DATA1 = []
    S_Data = []
    S_label = []
    cnt = 0
    cw_directory = os.getcwd()

  #  H_dataset = cw_directory + r'\Dataset2'

   # for filename in os.listdir(H_dataset):
        # Process each file in the directory
    #    pass
def handwriting_prediction(self, event=None):
        global T
        contents="Starting Handwriting based Dyslexia Prediction"
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)
        def compute_feats(image, kernels):
            feats = np.zeros((len(kernels), 2), dtype=np.double)
            for k, kernel in enumerate(kernels):
                filtered = nd.convolve(image, kernel, mode='wrap')
                feats[k, 0] = filtered.mean()
                feats[k, 1] = filtered.var()
            return feats

        def GLCM_Feature(cropped):
            # GLCM Feature extraction
            glcm = graycomatrix(cropped, [1, 2], [0, np.pi/2], levels=256, normed=True, symmetric=True)
            dissim = (graycoprops(glcm, 'dissimilarity'))
            dissim=np.reshape(dissim, dissim.size)
            correl = (graycoprops(glcm, 'correlation'))
            correl=np.reshape(correl,correl.size)
            energy = (graycoprops(glcm, 'energy'))
            energy=np.reshape(energy,energy.size)
            contrast= (graycoprops(glcm, 'contrast'))
            contrast= np.reshape(contrast,contrast.size)
            homogen= (graycoprops(glcm, 'homogeneity'))
            homogen = np.reshape(homogen,homogen.size)
            asm =(graycoprops(glcm, 'ASM'))
            asm = np.reshape(asm,asm.size)
            glcm = glcm.flatten()
            Mn=sum(glcm)
            Glcm_feature = np.concatenate((dissim,correl,energy,contrast,homogen,asm,Mn),axis=None)
            return Glcm_feature

        list1= ['Dyslexia Handwriting', 'Normal Handwriting']

            #Read Image
        #S_filename = filedialog.askopenfilename(title='Select Signature Image')
        #S_img = cv2.imread(S_filename)
        Sh_img=cv2.resize('temp.png',(300,50))
        cv2.imwrite('temp.png',Sh_img)        
        if len(S_img.shape) == 3:
            G_img = cv2.cvtColor(S_img, cv2.COLOR_RGB2GRAY)
        else:
            G_img=S_img.copy()

        load = Image.open("temp.png")
        logo_img = ImageTk.PhotoImage(load)     
        image1=Label(self, image=logo_img,borderwidth=2, highlightthickness=5, height=300, width=400, bg='white')
        image1.image = logo_img
        image1.place(x=50, y=120)
        
        cv2.imshow('Input Image',cv2.resize(G_img,(300,50)))
        cv2.waitKey(0)           
            #Gaussian Filter and thresholding image
        blur_radius = 2
        blurred_image = ndimage.gaussian_filter(G_img, blur_radius)
        threshold, binarized_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('Segmented Image',cv2.resize(binarized_image,(300,50)))
        cv2.waitKey(0)
            # Find the center of mass
        r, c = np.where(binarized_image == 0)
        r_center = int(r.mean() - r.min())
        c_center = int(c.mean() - c.min())

            # Crop the image with a tight box
        cropped = G_img[r.min(): r.max(), c.min(): c.max()]

            ## Signature Feature extraction
        Average,Percentage = pressure.pressure(cropped)
        top, middle, bottom = zones.findZone(cropped)

        Glcm_feature_signature =GLCM_Feature(cropped)
        Glcm_feature_signature=Glcm_feature_signature.flatten()

        bw_img,angle1= segmentation.Segmentation(G_img)

        feature_matrix1 = np.concatenate((Average,Percentage,10,top, middle, bottom,Glcm_feature_signature),axis=None)

        Model_lod1 = joblib.load("Trained_H_Model.pkl")

        #ypred = Model_lod.predict(cv2.transpose(Feature_matrix))
        pred=Model_lod1.predict(cv2.transpose(feature_matrix1))
        Dyslexia_writing=pred[0]
        print(pred)
        contents=list1[pred[0]]
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        st.write(contents)
        self.Dyslexia_writing=Dyslexia_writing


if __name__ == "__main__":
    main()
