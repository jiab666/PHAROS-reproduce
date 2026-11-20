import scanpy as sc


src = "/srv/pharos/figure6_bcc/output/malignant_bcc_log1p.SE600M.h5ad"
dst = "/srv/pharos/figure6_bcc/output/malignant_bcc_log1p.post_only.SE600M.h5ad"

a = sc.read_h5ad(src)
keep = ["resist_post", "response_post"]
a = a[a.obs["response_pre_post"].isin(keep)]

print("shape:", a.shape)
print(a.obs["response_pre_post"].value_counts())
a.write_h5ad(dst)
print("wrote", dst)
