#' @export
BlitzMLLassoRegression = R6::R6Class(
  "BlitzMLLassoRegression",
  public = list(
    initialize = function(loss = c("squared", "logistic"), #c("squared", "huber", "logistic", "squared_hinge", "smoothed_hinge"),
                          lambda = "auto",
                          n_lambda = 20,
                          lambda_min_fraction = 0.0001,
                          lambda_max_fraction = 0.3,
                          include_bias_term = TRUE,
                          tol = 1e-3,
                          max_time = 60,
                          max_iter = 10000,
                          log_dir = tempdir()) {

      loss = match.arg(loss)
      check_loss(loss)

      stopifnot(
        is.numeric(lambda_min_fraction) &&
        is.numeric(lambda_max_fraction) &&
        is.numeric(n_lambda) &&
        is.numeric(tol) &&
        is.numeric(max_time) &&
        is.numeric(max_iter) &&
        is.logical(include_bias_term)
      )

      stopifnot(is.numeric(lambda) || identical(lambda, "auto"))
      if(is.numeric(lambda)) lambda = sort(lambda)

      private$n_lambda = n_lambda[[1]]
      private$lambda_min_fraction = lambda_min_fraction[[1]]
      private$lambda_max_fraction = lambda_max_fraction[[1]]
      private$tol = tol
      private$max_iter = max_iter
      private$max_time = max_time
      private$loss = loss
      private$log_dir = log_dir

      private$params = list(lambda = lambda,
                            undefined_param = 0,
                            include_bias_term = as.numeric(include_bias_term),
                            loss_index = encode_loss(private$loss),
                            loss = private$loss)
      futile.logger::flog.debug("params : %s", paste(names(private$params), private$params, sep = "=", collapse = ", "))
    },
    fit = function(x, y, ...) {
      stopifnot(is.numeric(y))
      if(inherits(x, "sparseMatrix")) {
        x = as(x, "dgCMatrix")
      } else {
        if(! inherits(x, "matrix"))
          stop("'x' should inherit from 'matrix' or 'Matrix::sparseMatrix'")
      }

      stopifnot(nrow(x) == length(y))
      solver = BlitzML_create_solver(private$loss)

      # BlitzML requirement - positive classes encoded as 1 and negative as -1
      # FIXME this is for logistic solver only!
      if(private$loss == "logistic") y = ifelse(y > 0, 1, -1)

      BlitzML_set_tolerance(solver, private$tol)
      BlitzML_set_use_working_sets(solver, TRUE)
      BlitzML_set_max_time(solver, private$max_time)
      BlitzML_set_max_iterations(solver, private$max_iter)

      num_features = ncol(x)
      num_examples = nrow(x)
      # FIXME
      # https://github.com/tbjohns/BlitzML/blob/93f80a83e206001817a0b1a0b81db6455ef44a87/python-package/_sparse_linear.py#L141
      result_buffer = numeric(num_features + 1 + num_examples + 2)

      dataset = BlitzML_create_dataset(x, y)

      if(is.null(self$lambda_seq)) {
        if(identical(private$params$lambda, "auto")) {
          self$lambda_seq = private$generate_lambda_seq(x, y)
        } else {
          self$lambda_seq = private$params$lambda
        }
      }

      res = lapply(self$lambda_seq, function(lambda) {
        status_buffer = raw(STATUS_BUFFER_SIZE)
        start = Sys.time()
        params = c(lambda = lambda,
                   # FIXME https://github.com/tbjohns/BlitzML/blob/93f80a83e206001817a0b1a0b81db6455ef44a87/python-package/_sparse_linear.py#L134
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
      res = x %*% self$coef
      if(private$loss == "logistic") res = sigmoid(res)
      as.matrix(res)
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
      class(ret) = c("cv.BlitzMLLassoRegression", class(ret))
      # attr(ret, "coef") = lapply(res, function(x) attr(x, "coef"))
      ret
    },
    coef = NULL,
    lambda_seq = NULL
  ),
  private = list(
    generate_lambda_seq = function(x, y) {
      # lambda_max = find_max_lambda(x, y, params = c(0, 0, private$params$include_bias_term, private$params$loss_index))
      lambda_max = find_max_lambda(x, y, private$params)
      flog.info("found max lambda: %f", lambda_max)
      lambda_seq = seq(
        log10(private$lambda_min_fraction * lambda_max),
        log10(private$lambda_max_fraction * lambda_max),
        length.out = private$n_lambda
        )
      lambda_seq = 10 ** lambda_seq
      lambda_seq
    },
    params = NULL,
    loss = NULL,
    log_dir = NULL,
    lambda_min_fraction = NULL,
    lambda_max_fraction = NULL,
    n_lambda = NULL,
    tol = NULL,
    max_iter = NULL,
    max_time = NULL
  )
)

BlitzML_create_solver = function(loss) {
  check_loss(loss)
  switch (loss,
          squared = BlitzML_new_linear_solver(),
          huber = BlitzML_new_solver(),
          logistic = BlitzML_new_logreg_solver(),
          squared_hinge = BlitzML_new_solver(),
          smoothed_hinge = BlitzML_new_solver(),
          stop(sprintf("solver for '%s' loss function is not supported yet", loss))
  )
}

check_loss = function(loss) {
  if(!(loss %in% IMPLEMENTED_LOSS_FUNCTIONS))
    stop(sprintf("only %s loss supported at the moment", paste(paste("'", IMPLEMENTED_LOSS_FUNCTIONS, "'", sep = ""), collapse = "/")))
  invisible(TRUE)
}

encode_loss = function(x) {
  match(x, AVAILABLE_LOSS_FUNCTIONS) - 1L # loss codes start from 0 in BlitzML
}

find_max_lambda = function(x, y, params) {
  solver = BlitzML_create_solver(params$loss)
  dataset = BlitzML_create_dataset(x, y)
  # https://github.com/tbjohns/BlitzML/blob/93f80a83e206001817a0b1a0b81db6455ef44a87/python-package/_sparse_linear.py#L42
  params_vec = c(0, 0, params$include_bias_term, params$loss_index)
  params_vec = BlitzML_new_parameters(params_vec)

  BlitzML_solver_compute_max_l1_penalty(solver, dataset, params_vec)
}
