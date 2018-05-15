#' @useDynLib "blitzml", .registration=TRUE
#' @importFrom Rcpp sourceCpp
#' @import Matrix
#' @importFrom futile.logger flog.info flog.debug flog.trace

#' @export
LassoLogisticRegressionBlitzML = R6::R6Class(
  "LassoLogisticRegressionBlitzML",
  public = list(
    initialize = function(loss = c("squared", "huber", "logistic"),
                          lambda = "auto",
                          n_lambda = 20,
                          lambda_min_fraction = 0.0001,
                          lambda_max_fraction = 0.3,
                          include_bias_term = TRUE,
                          tol = 1e-3,
                          max_time = 60,
                          max_iter = 10000,
                          use_working_sets = TRUE,
                          log_dir = tempdir()) {
      if(loss != "logistic")
        stop("only 'logistic' loss supported at the moment")

      stopifnot(is.numeric(lambda) || identical(lambda, "auto"))
      if(is.numeric(lambda))
        lambda = sort(lambda)

      private$n_lambda = n_lambda[[1]]
      private$lambda_min_fraction = lambda_min_fraction[[1]]
      private$lambda_max_fraction = lambda_max_fraction[[1]]
      private$tol = tol
      private$max_iter = max_iter
      private$use_working_sets= use_working_sets
      private$max_time = max_time
      private$loss = match.arg(loss)
      private$log_dir = log_dir

      # loss_index:
      # 0 - squared
      # 1 - huber
      # 2 - logistic
      private$params = list(lambda = lambda,
                            undefined_param = 0,
                            include_bias_term = as.numeric(include_bias_term),
                            loss_index = match(private$loss, private$supported_losses) - 1L)
    },
    fit = function(x, y, ...) {
      stopifnot(is.numeric(y))
      stopifnot(inherits(x, "dgCMatrix"))
      stopifnot(nrow(x) == length(y))

      solver = switch (private$loss,
              logistic = BlitzML_new_sparse_logreg_solver(),
              stop(sprintf("solver for '%s' loss function is not supported yet", private$loss))
      )
      # BlitzML requirement - positive classes encoded as 1 and negative as -1
      # FIXME this is for logistic solver only!
      y = ifelse(y > 0, 1, -1)

      BlitzML_set_tolerance(solver, private$tol)
      BlitzML_set_use_working_sets(solver, private$use_working_sets)
      BlitzML_set_max_time(solver, private$max_time)
      BlitzML_set_max_iterations(solver, private$max_iter)

      num_features = ncol(x)
      num_examples = nrow(x)
      # FIXME
      # https://github.com/tbjohns/BlitzML/blob/93f80a83e206001817a0b1a0b81db6455ef44a87/python-package/_sparse_linear.py#L141
      result_buffer = numeric(num_features + 1 + num_examples + 2)

      dataset = BlitzML_new_sparse_dataset(x, y)

      if(is.null(self$lambda_seq)) {
        if(identical(private$params$lambda, "auto")) {
          self$lambda_seq = private$generate_lambda_seq(x, y)
        } else {
          self$lambda_seq = private$params$lambda
        }
      }

      res = lapply(self$lambda_seq, function(lambda) {
        status_buffer = raw(64)
        start = Sys.time()
        params = c(lambda = lambda,
                   undefined_param = private$params$undefined_param,
                   include_bias_term = private$params$include_bias_term,
                   loss_index = private$params$loss_index)
        params = BlitzML_new_parameters(params)
        BlitzML_solve_problem(solver, dataset, params, result_buffer, status_buffer, private$log_dir)

        coefs = result_buffer[1L:num_features]
        bias = result_buffer[[num_features + 1L]]

        status = rawToChar(status_buffer)
        time_spent =  difftime(Sys.time(), start, "sec")

        if(status == 'Exceeded time limit')
          flog.warn("haven't converged for lambda %f - exceeded time limit of %f sec", lambda, private$max_time)

        flog.trace("solved in %f sec for lambda %f, %d non-zero coef (status: '%s')",
                   time_spent, lambda, sum(coefs != 0), status)

        # put bias followed coefficients and convert to sparse matrix (actually vector)
        as(c(bias, coefs), "CsparseMatrix")
      })
      res = do.call(cbind, res)
      colnames(res) = paste("lambda=", self$lambda_seq, sep = "")
      cn = colnames(x)
      if(is.null(cn))
        cn = paste("V", seq_len(ncol(x)), sep = "")
      rownames(res) = c("bias", cn)
      self$coef = res
      invisible(self$coef)
    },
    predict = function(x, ...) {
      # add first dummy column of ones in order to add biases via matrix multiplication
      x = cbind(rep(1, nrow(x)), x)
      as.matrix(sigmoid(x %*% self$coef))
    },
    cross_validate = function(x, y,
                              fold_id = NULL,
                              n_folds = 4,
                              callbacks = list(auc = blitzml::auc),
                              cores = getOption("mc.cores", parallel::detectCores()),
                              ...) {

      if(is.null(self$lambda_seq) && identical(private$params$lambda, "auto")) {
        self$lambda_seq = private$generate_lambda_seq(x, y)
      }

      if (.Platform$OS.type != "unix" && cores > 1) {
        warning("Windows detected - setting mc.cores = 1")
        cores = 1L
      }

      lapply(callbacks, function(f) stopifnot(is.function(f) && length(formals(f)) == 2L))

      if(is.null(fold_id))
        fold_id = sample.int(n_folds, nrow(x), replace = TRUE)
      folds = unique(fold_id)

      res = parallel::mclapply(folds, function(fld) {
        i = fold_id == fld

        x_train = x[i, , drop = FALSE]
        y_train = y[i]

        x_test = x[!i, , drop = FALSE]
        y_test = y[!i]

        coef = self$fit(x_train, y_train)
        preds = self$predict(x_test)
        flog.trace("executing callbacks")
        scores = lapply(callbacks, function(f) {
          vapply(1:ncol(preds), function(col) {
            f(y_test, preds[, col])
          }, 0.0)
        })
        ret = data.frame(fold = fld, scores, list(lambda = self$lambda_seq))
        # attr(ret, "coef") = coef
        ret
      }, mc.cores = cores, ...)
      ret = do.call(rbind, res)
      class(ret) = c("cv.LassoLogisticRegressionBlitzML", class(ret))
      # attr(ret, "coef") = lapply(res, function(x) attr(x, "coef"))
      ret
    },
    coef = NULL,
    lambda_seq = NULL
  ),
  private = list(
    generate_lambda_seq = function(x, y) {
      lambda_max = find_max_lambda(x, y, params = c(0, 0, private$params$include_bias_term, private$params$loss_index))
      flog.info("found max lambda: %f", lambda_max)
      10**seq(log10(private$lambda_min_fraction * lambda_max),
              log10(private$lambda_max_fraction * lambda_max),
              length.out = private$n_lambda)
    },

    supported_losses = c("squared", "huber", "logistic"),
    params = NULL,
    loss = NULL,
    log_dir = NULL,
    lambda_min_fraction = NULL,
    lambda_max_fraction = NULL,
    n_lambda = NULL,
    tol = NULL,
    max_iter = NULL,
    max_time = NULL,
    use_working_sets = NULL
  )
)
