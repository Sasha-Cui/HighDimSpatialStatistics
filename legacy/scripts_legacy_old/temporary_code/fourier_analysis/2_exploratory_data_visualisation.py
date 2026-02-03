# Generated from /Users/cui/Documents/GitHub/HighDimSpatialStatistics/temporary_code/fourier_analysis/2_exploratory_data_visualisation.ipynb on 2026-02-03 10:30:29
# DO NOT EDIT: regenerate via scripts/tools/convert_notebooks.py

# %%
%run -i ~/project/preambles
%run -i ~/project/helper_functions
%run -i ~/project/fitting_functions

# %%
puck_list = ['Puck_230517_39'] # the largest puck
gene_list = [ "Inha", "Inhba", "Inhbb", "Fst", "Esr1", "Esr2", "Pgr", "Ar", "Cyp19a1",
    "Cyp17a1", "Cyp11a1", "Lhcgr", "Parm1", "Akr1c18", "Fshr", "Star", "Ptgfr",
    "Sfrp4", "Acvr1", "Acvr2a", "Acvr2b", "Ghr", "Lhb", "Cga"]
head = 0
adata, X,Y, gene_list=load_data(gene_list=gene_list, head=head, puck_list = puck_list)

# %%
import matplotlib.pyplot as plt
import numpy as np
# Function to plot spatial scatter plot of coordinates
def plot_coordinates(X):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], s=1, color='blue', alpha=0.5)
    plt.title("Spatial Distribution of Coordinates (X)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid()
    plt.show()

# Function to overlay a feature on the spatial distribution
def plot_feature_overlay(X, Y, feature_index=0):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X[:, 0], X[:, 1], c=Y[:, feature_index], cmap="viridis", s=1)
    plt.title(f"Feature {feature_index + 1} Overlay on Spatial Distribution")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.colorbar(sc, label=f"Feature {feature_index + 1} Value")
    plt.grid()
    plt.show()

# Function to plot multiple feature distributions
def plot_multiple_features(X, Y, feature_indices=[0, 1, 2]):
    num_features = len(feature_indices)
    plt.figure(figsize=(12, 4 * num_features))
    for i, feature_index in enumerate(feature_indices):
        plt.subplot(num_features, 1, i + 1)
        sc = plt.scatter(X[:, 0], X[:, 1], c=Y[:, feature_index], cmap="plasma", s=1)
        plt.title(f"Feature {feature_index} Overlay on Spatial Distribution")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.colorbar(sc, label=f"Feature {feature_index} Value")
        plt.grid()
    plt.tight_layout()
    plt.show()

# Visualizations
plot_coordinates(X)                 # Spatial scatter plot
plot_feature_overlay(X, Y, 0)       # Overlay first feature
plot_multiple_features(X, Y, [0, 1, 2])  # Overlay multiple features

# %%
import matplotlib.pyplot as plt
import numpy as np

# Example data for visualization
# Extract the second feature (Y[:, 1])
feature_values = Y[:, 1]

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(feature_values, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
plt.title("Distribution of Counts in Feature 1 (Y[:, 1])")
plt.xlabel("Feature Value")
plt.ylabel("Count")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# %%
feature_values.unique()

# %%
# Identify rows in Y where all values are zero
all_zero_rows = torch.all(Y == 0, dim=1)  # Boolean mask

# Count the number of such locations
zero_count = torch.sum(all_zero_rows).item()

# Print the result
print(f"Number of locations in X with all-zero gene expressions in Y: {zero_count}")

# %%
overall_number_of_cells = 0

for j in range(22):
    j_nonzero_rows = torch.sum(Y != 0, dim=1) == j  # Boolean mask for rows with exactly j nonzero value

    # Count the number of such locations
    j_nonzero_count = torch.sum(j_nonzero_rows).item()

    # Print the result
    print(f"Number of locations in X with exactly {j} nonzero gene expression(s) in Y: {j_nonzero_count}")
    overall_number_of_cells += j_nonzero_count

overall_number_of_cells

# %%
X.shape

# %%
Y.shape

# %%
for j in range(22):
    print("gene", j, "sum of expressions", sum(Y[:, j]).item())

# %%
nonzero_counts = []
for j in range(22):
    nonzero_count = torch.sum(Y[:, j] != 0).item()
    print("number of cells that see gene", j, ":", nonzero_count)
    nonzero_counts.append(nonzero_count/15907)

print("\ngenes which are found in more than 1/8 of the cells")
for j in range(22):
    nonzero_count = torch.sum(Y[:, j] != 0).item()
    if nonzero_count > 15907/8:
        print(j)
plt.hist(nonzero_counts, bins=20)

plt.hist(nonzero_counts, bins=20)

# %%
# # Think of it as a spatial point process

# %%
# Number of genes
num_genes = Y.shape[1]
num_genes

# %%
# Generate scatterplots for the first two genes (for testing)
for gene_idx in range(22):  # Adjust range for fewer plots
    valid_indices = (Y[:, gene_idx] > 0).nonzero(as_tuple=True)[0]
    X_filtered = X[valid_indices]
    Y_filtered = Y[valid_indices, gene_idx]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=Y_filtered, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(label='Expression Level')
    plt.title(f"Gene {gene_idx} Spatial Distribution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

# %%
[gene_list[i] for i in [6,9]]

# %%
[gene_list[i] for i in [0, 10, 15]]

# %%
# ### 1. **`Inha` (Inhibin Subunit Alpha)**
#
# - **Function**: 
#   - The **`Inha`** gene encodes the **alpha subunit** of the protein **inhibin**, a hormone involved in regulating reproductive physiology.
#   - Inhibin is a heterodimeric protein composed of an alpha subunit (encoded by `Inha`) and one of two beta subunits (encoded by other genes, like `Inhbb`).
#   - It is primarily produced in the gonads (ovaries and testes) and acts to inhibit the secretion of follicle-stimulating hormone (**FSH**) from the anterior pituitary gland.
#
# - **Biological Importance**:
#   - Plays a critical role in **feedback regulation** of the hypothalamic-pituitary-gonadal axis.
#   - Helps regulate ovarian follicle development and spermatogenesis.
#   - Abnormal expression or mutations in `Inha` have been linked to reproductive disorders and some cancers, including ovarian and adrenal tumors.
#
# ---
#
# ### 2. **`Cyp11a1` (Cytochrome P450 Family 11 Subfamily A Member 1)**
#
# - **Function**:
#   - The **`Cyp11a1`** gene encodes an enzyme also known as **cholesterol side-chain cleavage enzyme** or **P450scc**.
#   - This enzyme is a key player in the synthesis of steroid hormones, catalyzing the first step of steroidogenesis: the conversion of **cholesterol** into **pregnenolone**.
#
# - **Biological Importance**:
#   - Found in steroidogenic tissues, such as the adrenal glands, gonads, and placenta.
#   - Essential for the production of all steroid hormones, including glucocorticoids, mineralocorticoids, and sex steroids (e.g., estrogen, progesterone, testosterone).
#   - Disruption of `Cyp11a1` function can lead to adrenal insufficiency or disorders of steroid hormone synthesis.
#
# ---
#
# ### 3. **`Star` (Steroidogenic Acute Regulatory Protein)**
#
# - **Function**:
#   - The **`Star`** gene encodes the **steroidogenic acute regulatory protein**, which is critical for steroid hormone production.
#   - It facilitates the transport of cholesterol from the outer to the inner mitochondrial membrane, a necessary step for steroidogenesis.
#
# - **Biological Importance**:
#   - Found in steroid-producing cells of the adrenal glands and gonads.
#   - Works in conjunction with enzymes like **Cyp11a1** to ensure efficient steroid hormone synthesis.
#   - Mutations in `Star` can lead to congenital lipoid adrenal hyperplasia, a severe condition characterized by impaired steroid hormone production.
#
# ---
#
# ### Summary of Their Relationships:
# - **`Inha`** regulates reproductive hormone levels via inhibin, which can indirectly influence steroidogenesis.
# - **`Cyp11a1`** and **`Star`** are directly involved in the production of steroid hormones, with **`Star`** enabling cholesterol transport and **`Cyp11a1`** initiating steroidogenesis.
#
# Together, these genes are crucial for normal reproductive function and endocrine health. Their expression and regulation are tightly linked in physiological contexts such as puberty, pregnancy, and the stress response.

