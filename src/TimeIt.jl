module TimeIt

macro timeit(counter, ast::Expr)
    return quote
        runtime = @timed begin $(esc(ast)) end
        $(esc(counter)) += runtime.time
    end
end

export @timeit

end # module Timeit
