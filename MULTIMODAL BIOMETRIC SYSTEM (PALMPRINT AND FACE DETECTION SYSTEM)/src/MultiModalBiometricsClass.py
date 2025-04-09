
#Bukasa Muyombo, 
#Multimodal (face and palprint) biometrics
#one file
import cv2
import numpy as np
import os
import warnings

import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import seaborn as sns

class DataLoader:
    def __init__(self, palmprintFolder, faceFolder):
        self.palmprintFolder = palmprintFolder
        self.faceFolder = faceFolder

    def loadPalmprintData(self):
        palmprintData = []
        for filename in os.listdir(self.palmprintFolder):
            imagePath = os.path.join(self.palmprintFolder, filename)
            img = cv2.imread(imagePath)
            palmprintData.append(img)
        return palmprintData

    def loadFaceData(self):
        faceData = []
        for filename in os.listdir(self.faceFolder):
            imagePath = os.path.join(self.faceFolder, filename)
            img = cv2.imread(imagePath)
            faceData.append(img)
        return faceData

    def preprocess_data(self, data):
        resizedData = []
        for img in data:
            resizedIMg = cv2.resize(img, (256, 256))  # Resize to 256x256
            resizedData.append(resizedIMg)
        return resizedData


class FeatureExtractor:
    def __init__(self):
        self.base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.model = Model(inputs=self.base_model.input, outputs=GlobalAveragePooling2D()(self.base_model.output))

    def preprocessImage(self, image):
        resizedImage = cv2.resize(image, (224, 224))

        preprocessedImage = np.expand_dims(resizedImage, axis=0).astype('float32') / 255.0
        
        return preprocessedImage

    def extractPalmprintFeatures(self, palmprintImage):
        preprocessedImage = self.preprocessImage(palmprintImage)
        palmprintFeatures = self.model.predict(preprocessedImage)
        return palmprintFeatures

    def extract_facialeFeatures(self, facial_image):
        preprocessedImage = self.preprocessImage(facial_image)
        facialeFeatures = self.model.predict(preprocessedImage)
        return facialeFeatures
    



class SecondFeatureExtractor:
    def __init__(self):
        self.pca = PCA()

    def preprocessImage(self, image):
        resizedImage = cv2.resize(image, (224, 224))
        # Flatten and normalize the image
        ##resizedImage = cv2.resize(image, (224, 224))
        flattenedImage = resizedImage.flatten() / 255.0  
        return flattenedImage

    def extract_features(self, data):
        preprocessedData = [self.preprocessImage(img) for img in data]
        n_components = min(len(preprocessedData), preprocessedData[0].size)
        self.pca.n_components = n_components
        pcafeatures = self.pca.fit_transform(preprocessedData)
        return pcafeatures




####################################MATCHER CLASS FOR PIPELINE 1###############
class Matcher:
    def __init__(self):
        pass

    def matchpalmprintFeatures(self, features1, features2, method='euclidean'):
        if method == 'euclidean':
            euclideanDistance = np.linalg.norm(features1 - features2)
            similarityScore = 1 / (1 + euclideanDistance)
        elif method == 'cosine':
            similarityScore = 1 - cosine(features1, features2)
        else:
            raise ValueError("Invalid matching method. -------- Methd is not 'euclidean' or 'cosine'.")
        return similarityScore

    def matchFacialFeatures(self, features1, features2, method='euclidean'):
        if method == 'euclidean':
            euclideanDistance = np.linalg.norm(features1 - features2)
            #use either cosine or eucledian matching
            similarityScore = 1 / (1 + euclideanDistance)
        elif method == 'cosine':
            
            similarityScore = 1 - cosine(features1, features2)
        else:
            #
            raise ValueError("IInvalid matching method. -------- Methd is not 'euclidean' or 'cosine")
        return similarityScore

####################################MATCHER CLASS FOR PIPELINE 2######################    
class SecondMatcher:
    def __init__(self):
        pass

    def match_features(self, features1, features2):
        # Manhattan distance
        manhattandistance = np.sum(np.abs(features1 - features2))
        similarityScore = 1 / (1 + manhattandistance)
        
        return similarityScore


################################FUSIOIN CLASS FOR PIPELINE 1###################################
class Fusion:
    def scoreLevelFusion(self, scorePalmprint, scoreFace):
        return (scorePalmprint + scoreFace) / 2
    
    def featureLevelFusion(self, featuresPalmprint, featuresFace):
        featuresPalmprint = np.array(featuresPalmprint)
        featuresFace = np.array(featuresFace)
        
        if featuresPalmprint.ndim == 1:
            featuresPalmprint = np.expand_dims(featuresPalmprint, axis=0)
        if featuresFace.ndim == 1:
            featuresFace = np.expand_dims(featuresFace, axis=0)
        
        return np.concatenate((featuresPalmprint, featuresFace), axis=1)
    
    def decision_level_fusion(self, decisionPalmprint, decisionFace):
        if decisionPalmprint == decisionFace:
            return decisionPalmprint
        else:
            return "Undetermined"
        
    def match_level_fusion(self, scorePalmprint, scoreFace):
        # chose the maximum score
        return max(scorePalmprint, scoreFace)
        
########################################PIPELINE 2 FUSION CLASS##################################
class SecondFusion:
    def weighted_sum_fusion(self, scorePalmprint, scoreFace):
        # weighted sum fusion
        wp = 0.7
        weight_face = 0.3
        fs = (wp * scorePalmprint) + (weight_face * scoreFace)
        return fs
    
    #feature level fusoin
    def featureLevelFusion(self, featuresPalmprint, featuresFace):  
        if not featuresPalmprint or not featuresFace:
            # If either featuresPalmprint or featuresFace is empty, return an empty array
            return np.array([])
        
        featuresPalmprint = np.array(featuresPalmprint)
        featuresFace = np.array(featuresFace)
        
        if featuresPalmprint.ndim == 1:
            featuresPalmprint = np.expand_dims(featuresPalmprint, axis=0)
        if featuresFace.ndim == 1:
            featuresFace = np.expand_dims(featuresFace, axis=0)
        
        return np.concatenate((featuresPalmprint, featuresFace), axis=1)

#############################################DECISION MAKER CLASS####################################
class DecisionMaker:
    def __init__(self):
        self.thresholds = {}

    def decide(self, fusionResult):
        threshold = self.thresholds.get("threshold", 0.5)
        
        # make sure fusionResult is a numpy array
        fusionResult = np.array(fusionResult)
        
        # Apply the threshold 
        decisions = fusionResult >= threshold
        
        # Ensure each decision element is processed correctly
        return ["Authenticated" if decision else "Rejected" for decision in decisions.flatten()]
    
    def set_thresholds(self, thresholds):
        self.thresholds = thresholds



class PerformanceEvaluator:
    def calculateFAR(self, aScores, impScores):
        far = 0.0
        if len(aScores) > 0:
            max_authentic_score = np.max(aScores)
            far = len([score for score in impScores if score >= max_authentic_score]) / len(impScores)
        return far
    
    def calculateFRR(self, aScores, impScores):
        frr = 0.0
        if len(impScores) > 0:
            max_impostor_score = np.max(impScores)
            frr = len([score for score in aScores if np.any(score < max_impostor_score)]) / len(aScores)
        return frr
    
    def calculate_recall(self, aScores, impScores, threshold=0.5):
        authentic_decisions = np.array(aScores) >= threshold
        impostor_decisions = np.array(impScores) >= threshold
        
        true_authentic = np.count_nonzero(authentic_decisions)
        total_authentic = len(aScores)
        
        recall = true_authentic / total_authentic
        return recall

    def calculate_accuracy(self, aScores, impScores, threshold=0.5):
        authentic_decisions = np.array(aScores) >= threshold
        impostor_decisions = np.array(impScores) >= threshold
        
        true_authentic = np.count_nonzero(authentic_decisions)
        true_impostor = np.count_nonzero(~impostor_decisions)
        
        total_authentic = len(aScores)
        total_impostor = len(impScores)
        
        accuracy = (true_authentic + true_impostor) / (total_authentic + total_impostor)
        return accuracy


    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred, labels=["Authenticated", "Rejected"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Authenticated", "Rejected"], yticklabels=["Authenticated", "Rejected"])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()



class BiometricSystem:
    def __init__(self, palmprintFolder, faceFolder):
        self.dataLoader = DataLoader(palmprintFolder, faceFolder)
        self.feature_extractor = FeatureExtractor()
        self.matcher = Matcher()
        self.fusion = Fusion()
        self.decision_maker = DecisionMaker()
        self.performance_evaluattor = PerformanceEvaluator()
        self.aScores = []
        self.impScores = []
        self.enrPPFeats = []
        self.enrolled_facialeFeatures = []

    def enroll_user(self):
        palmprintData = self.dataLoader.loadPalmprintData()
        faceData = self.dataLoader.loadFaceData()
        
        palmprintFeatures = [self.feature_extractor.extractPalmprintFeatures(img) for img in palmprintData]
        facialeFeatures = [self.feature_extractor.extract_facialeFeatures(img) for img in faceData]
        
        self.enrPPFeats = palmprintFeatures
        self.enrolled_facialeFeatures = facialeFeatures
        print("User enrolled successfully.")

    def authenticate_user(self, palmprintImage, facial_image):
        palmprintFeatures = self.feature_extractor.extractPalmprintFeatures(palmprintImage)
        facialeFeatures = self.feature_extractor.extract_facialeFeatures(facial_image)
        
        pps = [self.matcher.matchpalmprintFeatures(palmprintFeatures, feat) for feat in self.enrPPFeats]
        fis = [self.matcher.matchFacialFeatures(facialeFeatures, feat) for feat in self.enrolled_facialeFeatures]
        
        fusion_score = self.fusion.featureLevelFusion(pps, fis)
        decision = self.decision_maker.decide(fusion_score)
        
        # Flatten the fusion_score array for storing as single values
        self.aScores.extend(fusion_score.flatten())
        
        # Use actual impostor images for realistic testing (placeholder for simplicity)
        impPPImage = np.random.rand(224, 224, 3)
        impFImage = np.random.rand(224, 224, 3)
        impFFeats = self.feature_extractor.extractPalmprintFeatures(impPPImage)
        impPeats = self.feature_extractor.extract_facialeFeatures(impFImage)
        impostor_pps = [self.matcher.matchpalmprintFeatures(impFFeats, feat) for feat in self.enrPPFeats]
        impostor_fis = [self.matcher.matchFacialFeatures(impPeats, feat) for feat in self.enrolled_facialeFeatures]
        ImpFS = self.fusion.featureLevelFusion(impostor_pps, impostor_fis)
        self.impScores.extend(ImpFS.flatten())
        
        print("User authenticated successfully.")
        return decision
    
    def authenticateUserMaLFusion(self, palmprintImage, facial_image):
        palmprintFeatures = self.feature_extractor.extractPalmprintFeatures(palmprintImage)
        facialeFeatures = self.feature_extractor.extract_facialeFeatures(facial_image)
        
        pps = [self.matcher.matchpalmprintFeatures(palmprintFeatures, feat) for feat in self.enrPPFeats]
        fis = [self.matcher.matchFacialFeatures(facialeFeatures, feat) for feat in self.enrolled_facialeFeatures]
        
        fusion_score = self.fusion.match_level_fusion(pps, fis)
        decision = self.decision_maker.decide(fusion_score)
        
        # Flatten the fusion_score array for storing as single values
        self.aScores.extend(fusion_score.flatten())
        
        # Use actual impostor images for realistic testing (placeholder for simplicity)
        impPPImage = np.random.rand(224, 224, 3)
        impFImage = np.random.rand(224, 224, 3)
        impFFeats = self.feature_extractor.extractPalmprintFeatures(impPPImage)
        impPeats = self.feature_extractor.extract_facialeFeatures(impFImage)
        impostor_pps = [self.matcher.matchpalmprintFeatures(impFFeats, feat) for feat in self.enrPPFeats]
        impostor_fis = [self.matcher.matchFacialFeatures(impPeats, feat) for feat in self.enrolled_facialeFeatures]
        ImpFS = self.fusion.featureLevelFusion(impostor_pps, impostor_fis)
        self.impScores.extend(ImpFS.flatten())
        
        print("Athentication process has ended=========.")
        return decision
    
    def authenticateUserFLFusion(self, palmprintImage, facial_image):
        palmprintFeatures = self.feature_extractor.extractPalmprintFeatures(palmprintImage)
        facialeFeatures = self.feature_extractor.extract_facialeFeatures(facial_image)
        
        pps = [self.matcher.matchpalmprintFeatures(palmprintFeatures, feat) for feat in self.enrPPFeats]
        fis = [self.matcher.matchFacialFeatures(facialeFeatures, feat) for feat in self.enrolled_facialeFeatures]
        
        fusion_score = self.fusion.featureLevelFusion(pps, fis)
        decision = self.decision_maker.decide(fusion_score)
        
        # Flatten the fusion_score array for storing as single values
        self.aScores.extend(fusion_score.flatten())
        
        # Use actual impostor images for realistic testing (placeholder for simplicity)
        impPPImage = np.random.rand(224, 224, 3)
        impFImage = np.random.rand(224, 224, 3)
        impFFeats = self.feature_extractor.extractPalmprintFeatures(impPPImage)
        impPeats = self.feature_extractor.extract_facialeFeatures(impFImage)
        impostor_pps = [self.matcher.matchpalmprintFeatures(impFFeats, feat) for feat in self.enrPPFeats]
        impostor_fis = [self.matcher.matchFacialFeatures(impPeats, feat) for feat in self.enrolled_facialeFeatures]
        ImpFS = self.fusion.featureLevelFusion(impostor_pps, impostor_fis)
        self.impScores.extend(ImpFS.flatten())
        
        print("User authenticated successfully.")
        return decision

        

    def evaluate_performance(self):
        far = self.performance_evaluattor.calculateFAR(self.aScores, self.impScores)
        frr = self.performance_evaluattor.calculateFRR(self.aScores, self.impScores)
        accuracy = self.performance_evaluattor.calculate_accuracy(self.aScores, self.impScores)
        recall = self.performance_evaluattor.calculate_recall(self.aScores, self.impScores)
        
        #PRINT METRICS
        print("False Acceptance Rate (FAR):", far)
        print("False Rejection Rate (FRR):", frr)

        print("Accuracy: ", accuracy)
        print("Recall: ", recall)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].hist(self.aScores, bins=20, color='blue', alpha=0.7)
        axes[0, 0].set_title('Authentic Scores')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(self.impScores, bins=20, color='red', alpha=0.7)
        axes[0, 1].set_title('Impostor Scores')
        axes[0, 1].set_xlabel('Score')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[1, 0].bar(['FAR', 'FRR'], [far, frr], color=['green', 'orange'])
        axes[1, 0].set_title('FAR vs FRR')
        axes[1, 0].set_ylabel('Rate')
        
        axes[1, 1].bar(['Accuracy', 'Recall'], [accuracy, recall], color=['purple', 'brown'])
        axes[1, 1].set_title('Accuracy vs Recall')
        axes[1, 1].set_ylabel('Rate')
        
        fig.suptitle("PIPELINE 1 STATISTICS", fontsize = 16)

        plt.tight_layout()
    
        plt.show(block = False)

    ##Make a confusion matrix
    def ConfusionMatrix(self):    
        y_true = ['Authenticated'] * len(self.aScores) + ['Rejected'] * len(self.impScores)
        y_pred = self.decision_maker.decide(self.aScores) + self.decision_maker.decide(self.impScores)
        self.performance_evaluattor.plot_confusion_matrix(y_true, y_pred)


###########################Biometric class to handle the seconf pipeline ##############
class SecondBiometricSystem:
    def __init__(self, palmprintFolder, faceFolder):
        self.dataLoader = DataLoader(palmprintFolder, faceFolder)
        self.secFeatExtractor = SecondFeatureExtractor()
        self.second_matcher = SecondMatcher()
        self.second_fusion = SecondFusion()  # Add this line
        self.decision_maker = DecisionMaker()
        self.performance_evaluattor = PerformanceEvaluator()
        self.aScores = []
        self.impScores = []
        self.enrPPFeats = []
        self.enrolled_facialeFeatures = []

    def enroll_user(self):
        # #Use SecondFeatureExtractor for enrollment
        palmprintData = self.dataLoader.loadPalmprintData()
        faceData = self.dataLoader.loadFaceData()
        palmprintFeatures = self.secFeatExtractor.extract_features(palmprintData)
        facialeFeatures = self.secFeatExtractor.extract_features(faceData)
        self.enrPPFeats = palmprintFeatures
        self.enrolled_facialeFeatures = facialeFeatures
        print("User enrolled successfully.")

    def authenticate_user(self, palmprintImage, facial_image):
        # Use the SecondFeatureExtractor for authentication
        palmprintFeatures = self.secFeatExtractor.extract_features([palmprintImage])
        facialeFeatures = self.secFeatExtractor.extract_features([facial_image])
        
        # Matching and fusion using SecondMatcher and SecondFusion class
        pps = [self.second_matcher.match_features(palmprintFeatures, feat) for feat in self.enrPPFeats]
        fis = [self.second_matcher.match_features(facialeFeatures, feat) for feat in self.enrolled_facialeFeatures]
        
        fusion_score = self.second_fusion.featureLevelFusion(pps, fis)  # Corrected fusion method call
        decision = self.decision_maker.decide(fusion_score)
        
        # Flatten the fusion_score array for storing as single values
        self.aScores.extend(fusion_score.flatten())
        
        # Use actual impostor images for realistic testing (placeholder for simplicity)
        impPPImage = np.random.rand(224, 224, 3)
        impFImage = np.random.rand(224, 224, 3)
        impFFeats = self.secFeatExtractor.extract_features([impPPImage])
        impPeats = self.secFeatExtractor.extract_features([impFImage])
        impostor_pps = [self.second_matcher.match_features(impFFeats, feat) for feat in self.enrPPFeats]
        impostor_fis = [self.second_matcher.match_features(impPeats, feat) for feat in self.enrolled_facialeFeatures]
        ImpFS = self.second_fusion.featureLevelFusion(impostor_pps, impostor_fis)
        self.impScores.extend(ImpFS.flatten())
        
        print("User authenticated successfully.")
        return decision
    
    

    def evaluate_performance(self):
        far = self.performance_evaluattor.calculateFAR(self.aScores, self.impScores)
        frr = self.performance_evaluattor.calculateFRR(self.aScores, self.impScores)
        
        accuracy = self.performance_evaluattor.calculate_accuracy(self.aScores, self.impScores)
        recall = self.performance_evaluattor.calculate_recall(self.aScores, self.impScores)
        
        
        print("False Acceptance Rate (FAR):", far)
        print("False Rejection Rate (FRR):", frr)

        print("Accuracy: ", accuracy)
        print("Recall: ", recall)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].hist(self.aScores, bins=20, color='blue', alpha=0.7)
        axes[0, 0].set_title('Authentic Scores')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(self.impScores, bins=20, color='red', alpha=0.7)
        axes[0, 1].set_title('Impostor Scores')
        axes[0, 1].set_xlabel('Score')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[1, 0].bar(['FAR', 'FRR'], [far, frr], color=['green', 'orange'])
        axes[1, 0].set_title('FAR vs FRR')
        axes[1, 0].set_ylabel('Rate')
        
        axes[1, 1].bar(['Accuracy', 'Recall'], [accuracy, recall], color=['purple', 'brown'])
        axes[1, 1].set_title('Accuracy vs Recall')
        axes[1, 1].set_ylabel('Rate')
        
        fig.suptitle("PIPELINE 2 STATISTICS", fontsize = 16)

        plt.tight_layout()
        
        #SHOW THE PLOT
        plt.show(block = False)
    

    def ConfusionMatrix(self):    
        y_true = ['Authenticated'] * len(self.aScores) + ['Rejected'] * len(self.impScores)
        y_pred = self.decision_maker.decide(self.aScores) + self.decision_maker.decide(self.impScores)
        self.performance_evaluattor.plot_confusion_matrix(y_true, y_pred)

        


palmprintFolder = "dataset/train/palm/1/"
faceFolder = "dataset/train/face/1/"

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    dataLoader = DataLoader(palmprintFolder, faceFolder)

    palmprintData = dataLoader.loadPalmprintData()
    faceData = dataLoader.loadFaceData()
    preprocessedpalmprintData = dataLoader.preprocess_data(palmprintData)
    preprocessedFaceData = dataLoader.preprocess_data(faceData)

    feature_extractor = FeatureExtractor()
    palmprintImage = cv2.imread('dataset/test/palm/0001_0001.bmp')
    facial_image = cv2.imread('dataset/test/face/1(1).jpg')

    palmprintFeatures = feature_extractor.extractPalmprintFeatures(palmprintImage)
    facialeFeatures = feature_extractor.extract_facialeFeatures(facial_image)

    matcher = Matcher()
    facialeFeatures1 = np.array([0.5, 0.6, 0.7, 0.8])
    facialeFeatures2 = np.array([0.6, 0.7, 0.8, 0.9])
    palmprintFeatures1 = np.array([0.1, 0.2, 0.3, 0.4])
    palmprintFeatures2 = np.array([0.2, 0.3, 0.4, 0.5])
    

    palmprint_similarity_euclidean = matcher.matchpalmprintFeatures(palmprintFeatures1, palmprintFeatures2)
    facial_similarity_euclidean = matcher.matchFacialFeatures(facialeFeatures1, facialeFeatures2)
    print("\n","-------------MODEL STATISTICS---------------", "\n")

    print("Palmprint similarity (Euclidean):", palmprint_similarity_euclidean)
    print("Facial similarity (Euclidean):", facial_similarity_euclidean)

    print("\n","------------PIPELINE 1 STATISTICS------------", "\n")

    biometricSystem = BiometricSystem(palmprintFolder, faceFolder)
    biometricSystem.enroll_user()

    authentication_result = biometricSystem.authenticate_user(palmprintImage, facial_image)
    print("\n","Authentication Feature level result:", authentication_result, "\n")


    authentication_result_fl = biometricSystem.authenticate_user(palmprintImage, facial_image)
    print("\n\n","Authentication  Score Level result:", authentication_result_fl, "\n")
    

    #authentication_result_ml = biometricSystem.authenticateUserMaLFusion(palmprintImage, facial_image)
    #print("\n\n","Authentication match Fusion Level:", authentication_result_ml, "\n")

    
    biometricSystem.evaluate_performance()
    input("Press any key to exit...")
    biometricSystem.ConfusionMatrix()

    print("\n","------------PIPELINE 2 STATISTICS------------", "n")

    # Instantiate the second biometric system
    secondBioSystem = SecondBiometricSystem(palmprintFolder, faceFolder)

    

    # Enroll users for the second pipeline
    secondBioSystem.enroll_user()
    print("Enrollment completed for SecondBiometricSystem.")

    # Authentication Process for SecondBiometricSystem
    authentication_result_second = secondBioSystem.authenticate_user(palmprintImage, facial_image)
    print("Authentication result for SecondBiometricSystem:", authentication_result_second)

    # Evaluate performance for the second pipeline
    secondBioSystem.evaluate_performance()
    input("Press any key to exit...")    
    secondBioSystem.ConfusionMatrix()
    print("Performance evaluation for SecondBiometricSystem completed.")
    input("Press any key to exit...")
