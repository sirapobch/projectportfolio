================================================================================
README — NECO Individual Project: Real vs AI-Generated Image Classification
MLP (HOG features) vs CNN (end-to-end)
By Sirapob Charoenchaipiyakul
================================================================================

OVERVIEW
--------
This test script (Updated_Test_script.ipynb) evaluates two pre-trained models —
an MLP operating on HOG features and a CNN operating on raw images — against a
held-out test set drawn from the GRAVEX-200K dataset. It reproduces all
evaluation figures and metrics reported in the project report.


--------------------------------------------------------------------------------
DATASET DOWNLOAD
--------------------------------------------------------------------------------
The test set is too large to be included in the submission zip file.
Please download the full dataset folder from the following Google Drive link:

    https://drive.google.com/drive/folders/1rgC2d4N4arifQ1dC-Xa2ziss-BziDhyz?usp=sharing

The download contains the complete folder structure required by the test script
(see "Required Directory Layout" below). Download and place it inside the
project root before running the notebook.


--------------------------------------------------------------------------------
REQUIRED DIRECTORY LAYOUT
--------------------------------------------------------------------------------
After downloading, the project root must look exactly like this.
Launch Jupyter from inside the top-level "NECO Project/" folder.

    NECO Project/
    ├── Test_script.ipynb     <- this notebook
    ├── Dataset/
    │   ├── test_labels.csv
    │   └── my_real_vs_ai_dataset/
    │       ├── ai_images/            <- AI-generated test images
    │       └── real/                 <- real test images
    └── outputs/
        ├── HPC_best_mlp_final.pth       <- trained MLP weights
        ├── HPC_best_cnn_final.pth       <- trained CNN weights
        ├── HPC_best_configs.json        <- best hyperparameter configs (JSON)
        ├── HPC_hog_scaler.pkl           <- StandardScaler fitted on training HOG features
        └── test_prep/
            └── HPC_test_hog_features.npz  <- pre-extracted HOG features for the test set

All files prefixed with "HPC_" were produced during training on HPC and must
be present in the outputs/ folder before running the test script. These are
included in the Google Drive folder linked above.


--------------------------------------------------------------------------------
SOFTWARE REQUIREMENTS
--------------------------------------------------------------------------------
Python version : 3.10 or higher (PyTorch 2.x requires Python ≥ 3.10)
Notebook runner: Jupyter Notebook or JupyterLab

Recommended: run inside a virtual environment (venv or conda).


--------------------------------------------------------------------------------
REQUIRED PYTHON LIBRARIES & VERSIONS
--------------------------------------------------------------------------------
Install all dependencies with:

    pip install torch==2.7.0 torchvision==0.22.0 numpy pandas pillow \
                scikit-image scikit-learn matplotlib seaborn

It is recommended to pin torch and torchvision to the versions below rather
than installing the latest, as newer major releases may introduce breaking
API changes. The versions listed were stable and widely tested at the time
of submission.

Specific version requirements:

    Library          Recommended        Notes
    ---------------  -----------------  ----------------------------------------
    torch            2.7.0              Do NOT use 2.11+ without testing;
                                        weights_only=True is supported from 2.0+
    torchvision      0.22.0             Must be paired with torch 2.7.0
    numpy            1.23.0
    pandas           1.5.0
    Pillow           9.0.0              PIL.Image used for image loading
    scikit-image     0.19.0             skimage.feature.hog for HOG extraction
    scikit-learn     1.1.0              StandardScaler, metrics
    matplotlib       3.6.0
    seaborn          0.12.0

GPU is optional. The script automatically detects CUDA and falls back to CPU.
CPU inference is sufficient for the test set evaluation.


--------------------------------------------------------------------------------
HOW TO RUN THE TEST SCRIPT
--------------------------------------------------------------------------------
1. Download the dataset from the Google Drive link above and place it
   in the correct directory layout (see above).

2. Ensure all required libraries are installed (see above).

3. Launch Jupyter from inside the "NECO Project/" root directory:

       cd "NECO Project"
       jupyter notebook

   or, for JupyterLab:

       jupyter lab

4. Open "Updated_Test_script.ipynb" in the Jupyter interface.

5. Run all cells in order from top to bottom:
       Kernel → Restart & Run All
   or run each cell manually with Shift+Enter.

6. All evaluation figures (confusion matrices, metric bar chart, ROC curves,
   probability distributions, CNN misclassified examples, filter visualisation,
   HOG visualisation) and the final results summary table will be displayed
   inline. No files are written to disk by the test script.


--------------------------------------------------------------------------------
EXPECTED OUTPUTS
--------------------------------------------------------------------------------
The notebook produces the following inline outputs when run successfully:

    Cell 7  : MLP test accuracy and weighted F1
    Cell 8  : CNN test accuracy and weighted F1
    Cell 9  : Per-class classification reports (precision, recall, F1)
    Cell 10 : Figure 3 — Confusion matrices (MLP and CNN side by side)
    Cell 11 : Figure 4 — Metric comparison bar chart
    Cell 12 : ROC curves with AUC values
    Cell 13 : Predicted probability distribution histograms
    Cell 14 : CNN misclassified example images (random sample of 10)
    Cell 15 : CNN first-layer learned filter visualisation
    Cell 16 : HOG visualisation on correctly-classified test examples
    Cell 17 : Final results summary table


--------------------------------------------------------------------------------
NOTES & TROUBLESHOOTING
--------------------------------------------------------------------------------
- Marginal numerical differences between the training notebook and this test
  script are expected and arise from floating-point precision differences
  in scaler serialisation across separate evaluation sessions. This is normal.

- Random seed 42 is fixed at the top of the notebook to ensure the same
  misclassified examples are shown each run.

- If you see a FileNotFoundError for an image, verify that the ai_images/
  and real/ folders are placed exactly as shown in the directory layout above.

- If torch.load raises a warning about weights_only, ensure your PyTorch
  version is 2.0.0 or higher.

- num_workers=0 is set for the CNN DataLoader to avoid multiprocessing issues
  on Windows. Increase to 2–4 on Linux/macOS for faster data loading.

================================================================================
