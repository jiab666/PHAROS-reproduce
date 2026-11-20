for (pkg in c("Matrix", "SeuratObject", "SingleCellExperiment", "SummarizedExperiment")) {
  cat(pkg, as.character(packageVersion(pkg)), "\n")
}
