# Neural-Nets-Final-Project
### Project by Kunal Patel, Nathan Pilbrough, Priscilla Cheng
### UCLA EE 239AS: Neural Networks (by Prof. Jonathon Kao)

#### Python 2.7.12  

#### Dataset  
  -Witheld for confidentiality purposes and owner rights  

#### Library Dependencies  
  Execute "pip install -r requirements.tct" within your virtual environment  
 
#### Code Execution  
  All models are implemented within the following files  
    -RecurrentModels.ipynb  
    -BraindecodeTest.ipynb  
    -DeepConvNet.ipynb  
    -DeepConvNetAll.ipynb  
    -DeepConvNetSTFT.ipynb  
  Implemented functions are found in UtilNNDL.py  

#### Save Data Naming Standard (./Best Models/Variables)  
  hist.History dictionaries:  
  [modelname]\_hist\_[SubjectName].npy  
    
  test_accuracy variable:  
  [modelname]\_testacc\_[SubjectName].npy  
  
  Confusion_matrix variable:  
  [modelname]\_conf\_[SubjectName].npy  
  
  Avoid underscores in modelname, makes the information easier to tokenize if we need to in the future  
  SubjectName examples = 'All', 'A10T', etc.  

#### Resources:  
https://kupdf.com/download/long-short-term-memory-networks-with-python_5a43310ce2b6f5d926656665_pdf  
http://www.bbci.de/competition/iv/desc_2a.pdf  
https://arxiv.org/pdf/1802.00308.pdf - ChronoNet  
