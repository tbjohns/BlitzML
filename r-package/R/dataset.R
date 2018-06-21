BlitzML_create_dataset = function(x, y) {

  if(!is.numeric(y)) stop("'y' should be numeric vector")

  if(!(inherits(x, "matrix") || inherits(x, "dgCMatrix")))
    stop("'x' should inherit from 'matrix' or 'dgCMatrix'")

  if(inherits(x, "dgCMatrix")) {
    BlitzML_new_sparse_dataset(x, y)
  } else {
    BlitzML_new_dense_dataset(x, y)
  }
}
