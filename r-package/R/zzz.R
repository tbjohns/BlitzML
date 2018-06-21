#' @useDynLib "blitzml", .registration=TRUE
#' @importFrom Rcpp sourceCpp
#' @import Matrix
#' @importFrom futile.logger flog.info flog.debug flog.trace

STATUS_BUFFER_SIZE = 64L

AVAILABLE_LOSS_FUNCTIONS = c("squared", "huber", "logistic", "squared_hinge", "smoothed_hinge")
IMPLEMENTED_LOSS_FUNCTIONS = c("squared", "logistic")
