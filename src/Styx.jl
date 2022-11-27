module Styx

################# using statements ################
using DataStructures
using Base: Base.Order.Ordering, Base.Order.Forward, Base.Order.ForwardOrdering

################# node defs ################
abstract type Node{V, S} end

abstract type AbstractSource{ID, V, S} <: Node{V, S} end

abstract type AbstractComputation{V, S} <: Node{V, S} end

struct NoLink{T <: Node, V, S} <: Node{V, S} end

struct FlowSource{ID, V} <: AbstractSource{ID, V, Missing} end

struct NoFlowSource{ID, V} <: AbstractSource{ID, V, Missing} end

struct CumulativeSum{T <: Node{<:Real}, V <: Real} <: AbstractComputation{V, Missing} end

struct CumulativeProduct{T <: Node{<:Real}, V <: Real} <: AbstractComputation{V, Missing} end

struct Sum{LHS <: Node{<:Real}, RHS <: Node{<:Real}, V <: Real} <: AbstractComputation{V, Missing} end

struct Difference{LHS <: Node{<:Real}, RHS <: Node{<:Real}, V <: Real} <: AbstractComputation{V, Missing} end

struct Product{LHS <: Node{<:Real}, RHS <: Node{<:Real}, V <: Real} <: AbstractComputation{V, Missing} end

struct Quotient{LHS <: Node{<:Real}, RHS <: Node{<:Real}} <: AbstractComputation{Float64, Missing} end

struct Previous{T <: Node, V} <: AbstractComputation{V, V} end

struct Buffer{T <: Node, Size, V} <: AbstractComputation{CircularBuffer{V}, V} end

struct Collector{T <: Node, V} <: AbstractComputation{Vector{V}, Missing} end

struct ConditionalCollector{T <: Node, U <: Function, V} <: AbstractComputation{Vector{V}, Missing} end

struct SimpleMA{B <: Buffer{<:Node{<:Real}}, S} <: AbstractComputation{Float64, S} end

struct SimpleEMA{T <: Node{<: Real}, Alpha} <: AbstractComputation{Float64, Missing} end

struct Maximum{T <: Node, O <: Ordering, V} <: AbstractComputation{V, Missing} end

struct Minimum{T <: Node, O <: Ordering, V} <: AbstractComputation{V, Missing} end

struct Negative{T <: Node{<:Real}, V} <: AbstractComputation{V, Missing} end

struct Abs{T <: Node{<:Real}, V} <: AbstractComputation{V, Missing} end

struct Counter{T <: Node} <: AbstractComputation{Int64, Missing} end

struct Pow{T <: Node{<:Real}, P} = AbstractComputation{Float64, Missing} end


################# node constructors ################
NoLink(::T) where T = NoLink{T, T |> valtype, T |> statetype}()

FlowSource(id::Int64, ::Type{V}) where V = FlowSource{id, V}()

FlowSource(::Type{V}) where V = FlowSource(Int64 |> rand, V)

NoFlowSource(id::Int64, ::Type{V}) where V = NoFlowSource{id, V}()

NoFlowSource(::Type{V}) where V = NoFlowSource(Int64 |> rand, V)

CumulativeSum(::T) where T = CumulativeSum{T, T |> valtype}()

CumulativeProduct(::T) where T = CumulativeProduct{T, T |> valtype}()

Sum(::LHS, ::RHS) where {LHS, RHS} = Sum{LHS, RHS, promote_type(LHS |> valtype, RHS |> valtype)}()

Difference(::LHS, ::RHS) where {LHS, RHS} = Difference{LHS, RHS, promote_type(LHS |> valtype, RHS |> valtype)}()

Product(::LHS, ::RHS) where {LHS, RHS} = Product{LHS, RHS, promote_type(LHS |> valtype, RHS |> valtype)}()

Quotient(::LHS, ::RHS) where {LHS, RHS} = Quotient{LHS, RHS}()

Previous(::T) where T = Previous{T, T |> valtype}()

Delta(n::T) where T = Difference(n, Previous(n))

Buffer(::T, size::Int64) where T = Buffer{T, size, T |> valtype}()

Collector(::T) where T = Collector{T, T |> valtype}()

ConditionalCollector(::T, ::U) where {T, U} = ConditionalCollector{T, U, T |> valtype}()

SimpleMA(n::T, size::Int64) where T = SimpleMA{Buffer(n, size) |> typeof, T |> valtype}()

SimpleEMA(::T, alpha::Float64) where T = SimpleEMA{T, alpha}()

Maximum(::T, ::O) where {T, O} = Maximum{T, O, T |> valtype}()

Maximum(n::T) where T = Maximum(n, Forward)

Minimum(::T, ::O) where {T, O} = Minimum{T, O, T |> valtype}()

Minimum(n::T) where T = Minimum(n, Forward)

MaxDrawDown(n::T) where T <: Node{<:Real} = Maximum(
    Difference(
        Maximum(n), 
        n
    )
)

Negative(::T) where T = Negative{T, T |> valtype}()

Abs(::T) where T = Abs{T, T |> valtype}()

Counter(::T) where T = Counter{T}()

Mean(n::T) where T <: Node{<:Real} = CumulativeSum(n) / Counter(n)

Square(n::T) where T = Product(n, n)

Variance(n::T) where T <: Node{<:Real} = Mean(Square(n)) - Square(Mean(n))

Pow(::T, p::Real) where T = Pow{T, p}()

StdDev(n::T) where T <: Node{<:Real} = Pow(
    Variance(n), 
    0.5
)


################ node calcs ################
calc(n::CumulativeSum{T}) where T = begin
    if is_val_init(T)
        setval!(n, getval(n, zero(n |> valtype)) + getval(T))
    end
    nothing
end

calc(n::CumulativeProduct{T}) where T = begin
    if is_val_init(T)
        setval!(n, getval(n, one(n |> valtype)) * getval(T))
    end
    nothing
end

calc(n::Sum{LHS, RHS}) where {LHS, RHS} = begin
    if is_val_init(LHS) && is_val_init(RHS)
        setval!(n, getval(LHS) + getval(RHS))
    end
    nothing
end

calc(n::Difference{LHS, RHS}) where {LHS, RHS} = begin
    if is_val_init(LHS) && is_val_init(RHS)
        setval!(n, getval(LHS) - getval(RHS))
    end
    nothing
end

calc(n::Product{LHS, RHS}) where {LHS, RHS} = begin
    if is_val_init(LHS) && is_val_init(RHS)
        setval!(n, getval(LHS) * getval(RHS))
    end
    nothing
end

calc(n::Quotient{LHS, RHS}) where {LHS, RHS} = begin
    if is_val_init(LHS) && is_val_init(RHS)
        setval!(n, getval(LHS) / getval(RHS))
    end
    nothing
end

calc(n::Previous{T}) where T = begin
    if is_val_init(T) && !is_state_init(n)
        current_val = T |> getval
        setstate!(n, current_val)
    elseif is_val_init(T)
        prev_val = n |> getstate
        current_val = T |> getval
        setstate!(n, current_val)
        setval!(n, prev_val)
    end
    nothing
end

calc(n::Buffer{T, Size}) where {T, Size} = begin
    if !is_val_init(n) && is_val_init(T)
        buffer = (n |> valtype)(Size)
        setval!(n, buffer)
    end
    if is_val_init(T)
        buffer = n |> getval
        next_val = T |> getval
        if isfull(buffer)
            setstate!(n, buffer[1])
        end
        push!(buffer, next_val)
        setval!(n, buffer)
    end
    nothing
end

calc(n::Collector{T}) where T = begin
    if !is_val_init(n) && is_val_init(T)
        setval!(n, (n |> valtype)())
    end
    if is_val_init(T)
        v = n |> getval
        next_val = T |> getval
        push!(v, next_val)
        setval!(n, v)
    end
    nothing
end

calc(n::ConditionalCollector{T, U}) where {T, U} = begin
    if !is_val_init(n) && is_val_init(T)
        setval!(n, (n |> valtype)())
    end
    if is_val_init(T)
        v = n |> getval
        next_val = T |> getval
        condition = U.instance
        if condition(next_val)
            push!(v, next_val)
        end
        setval!(n, v)
    end
    nothing
end

calc(n::SimpleMA{B}) where B <: Buffer{T, Size} where {T, Size} = begin
    if is_val_init(T) && is_state_init(B)
        sum = getstate(n, n |> statetype |> zero)
        next_val = T |> getval
        popped_val = B |> getstate
        sum = sum + next_val - popped_val
        setstate!(n, sum)
        setval!(n, sum / Size)
    elseif is_val_init(T) && is_val_init(B)
        sum = getstate(n, n |> statetype |> zero)
        next_val = T |> getval
        buffer = B |> getval
        sum = sum + next_val
        setstate!(n, sum)
        setval!(n, sum / length(buffer))
    end
    nothing
end

calc(n::SimpleEMA{T, Alpha}) where {T, Alpha} = begin
    if is_val_init(T)
        ema = getval(n, 0.0)
        next_val = T |> getval
        ema = (1 - Alpha)ema + (Alpha)next_val
        setval!(n, ema)
    end
    nothing
end

calc(n::Maximum{T, O}) where {T, O} = begin
    if !is_val_init(n) && is_val_init(T)
        val = T |> getval
        setval!(n, val)
    elseif is_val_init(T)
        val = T |> getval
        max_val = n |> getval
        if Base.Order.lt(O(), max_val, val)
            setval!(n, val)
        end
    end
    nothing
end

calc(n::Minimum{T, O}) where {T, O} = begin
    if !is_val_init(n) && is_val_init(T)
        val = T |> getval
        setval!(n, val)
    elseif is_val_init(T)
        val = T |> getval
        min_val = n |> getval
        if Base.Order.lt(O(), val, min_val)
            setval!(n, val)
        end
    end
    nothing
end

calc(n::Negative{T}) where T = begin
    if is_val_init(T)
        val = T |> getval
        setval!(n, -val)
    end
    nothing
end

calc(n::Abs{T}) where T = begin
    if is_val_init(T)
        setval!(n, T |> getval |> abs)
    end
    nothing
end

calc(n::Counter{T}) where T = begin
    if !is_val_init(n) && is_val_init(T)
        setval!(n, 1)
    elseif is_val_init(T)
        setval!(n, (n |> getval) + 1)
    end
    nothing
end

calc(n::Pow{T, P}) where {T, P} = begin
    if is_val_init(T)
        val = T |> getval
        pow = convert(Float64, val^P)
        setval!(n, pow)
    end
    nothing
end


################ default values for types to keep type stability #################
default(::Type{T}) where T <: Real = zero(T)

default(::Type{String}) = ""

default(::Type{Vector{T}}) where T = Vector{T}()

default(::Type{CircularBuffer{T}}) where T = CircularBuffer{T}(0)

default(::Type{T}) where T <: Tuple = map(default, T |> fieldtypes)

default(::Type{Missing}) = missing

default(::Type{Nothing}) = nothing

default(::Type{Bool}) = false

default(::Type{Char}) = '\0'

default(::Type{T}) where T <: Function = T.instance


############### node type introspection valtype/statetype #################
valtype(::Type{T}) where T <: Node{V, S} where {V, S} = V

statetype(::Type{T}) where T <: Node{V, S} where {V, S} = S

valtype(::Node{V, S}) where {V, S} = V

statetype(::Node{V, S}) where {V, S} = S


############## empty definitions of metaprogramming defined functions #################
getsource(_) = error("Undefined method")

setsource!(_, _) = error("Undefined method")

getval(_) = error("Undefined method")

getval(_, _) = error("Undefined method")

setval!(_, _) = error("Undefined method")

getstate(_) = error("Undefined method")

getstate(_, _) = error("Undefined method")

setstate!(_, _) = error("Undefined method")

is_val_init(_) = error("Undefined method")

is_state_init(_) = error("Undefined method")

flow!(_, _) = error("Undefined method")


##################### expanding and ordering nodes #######################
expand_and_order_nodes(::Type{T}) where T <: Tuple = begin
    expanded_type = T |> expand_nodes
    in_graph, out_graph, sources = expanded_type |> init_type_graph
    order_nodes(in_graph, out_graph, sources)
end

expand_nodes(::Type{T}) where T <: Tuple = begin
    node_types = Set{DataType}()
    for node_type in fieldtypes(T)
        node_types = union!(node_types, expand_node(node_type))
    end
    Tuple{node_types...}
end

expand_node(::Type{T}) where T <: Node = begin
    v = Vector{DataType}()
    push!(v, T)
    for type_param in T.parameters
        if type_param isa Type && type_param <: Node 
            push!(v, type_param)
            append!(v, expand_node(type_param))
        end
    end
    v
end

init_type_graph(::Type{T}) where T <: Tuple = begin
    in_graph = Dict(t => Set{DataType}() for t in fieldtypes(T))
    out_graph = Dict(t => Set{DataType}() for t in fieldtypes(T))
    sources = Set{DataType}()
    for node_type in fieldtypes(T)
        if node_type <: AbstractSource 
            push!(sources, node_type) 
        end
        for type_param in node_type.parameters
            if type_param isa Type && (type_param <: NoLink || type_param <: NoFlowSource)
                continue
            elseif type_param isa Type && type_param <: Node
                push!(out_graph[type_param], node_type)
                push!(in_graph[node_type], type_param)
            end
        end
    end
    in_graph, out_graph, sources
end

# Multi-source topological sort
order_nodes(in_graph::Dict{DataType, Set{DataType}}, out_graph::Dict{DataType, Set{DataType}}, sources::Set{DataType}) = begin    
    # Setup
    top_sort_stacks = Dict(t => Vector{DataType}() for t in sources)
    top_sorts = Dict(t => Vector{DataType}() for t in sources)

    # Inititalize stacks
    for source in sources
        push!(top_sort_stacks[source], source)
    end

    # Run algorithm
    while !all_empty(top_sort_stacks)
        # Set up state to keep track of which touched nodes correspond to which source
        touched_nodes = Dict(t => Vector{DataType}() for t in sources)
        
        # Add nodes to top_sorts and remove incoming edges on their neighbors
        for source in sources 
            stack = top_sort_stacks[source]
            if !isempty(stack)
                u = pop!(stack)
                push!(top_sorts[source], u)
                for v in out_graph[u]
                    safe_pop!(in_graph[v], u)
                    push!(touched_nodes[source], v)
                end
            end
        end

        # Add nodes with zero degree to stacks of corresponding source 
        for source in sources
            for v in touched_nodes[source]
                if isempty(in_graph[v])
                    push!(top_sort_stacks[source], v)
                end
            end
        end
    end

    # Return map of source types to ordered tuple type
    Dict(t => Tuple{top_sorts[t]...} for t in sources)
end

all_empty(d::Dict) = begin
    for v in values(d)
        if !isempty(v) return false end
    end
    return true
end

safe_pop!(s::Set{T}, ele::T) where T = begin
    ele in s ? pop!(s, ele) : ele
end


#################### hash types to string for use in metaprogramming variable names #####################
node_type_to_str(::Type{T}) where T <: Node = begin
    replace_left_brace = s -> replace(s, "{" => "_")
    replace_right_brace = s -> replace(s, "}" => "_")
    replace_comma = s -> replace(s, "," => "_")
    replace_space = s -> replace(s, " " => "_")
    replace_dot = s -> replace(s, "." => "_")
    replace_left_paren = s -> replace(s, "(" => "_")
    replace_right_paren = s -> replace(s, ")" => "_")
    string(T) |> replace_left_brace |> replace_right_brace |> replace_comma |> replace_space |> replace_dot |> replace_left_paren |> replace_right_paren
end

node_tuple_type_to_str(::Type{T}) where T <: Tuple = begin
    join(map(node_type_to_str, T |> fieldtypes), "__")
end

comp_desc_to_str(comp_desc::Dict{DataType, DataType}) = begin
    join(map(node_tuple_type_to_str, [values(comp_desc)...]), "___")
end

##################### implicitly generate required methods ########################
generate_get_source(comp_desc::Dict{DataType, DataType}) = begin
    global_flowing_source_var_name = Symbol("__source_", comp_desc |> comp_desc_to_str)
    flow_source_types = [t for t in keys(comp_desc) if t <: FlowSource]
    node_types = Set{DataType}()
    for tuple_type in values(comp_desc)
        node_types = union!(node_types, tuple_type |> fieldtypes)
    end

    quote
        const $global_flowing_source_var_name = Ref{Union{$(flow_source_types...)}}($((flow_source_types |> rand)()))

        getsource(::T) where T <: Union{$(node_types...)} = $global_flowing_source_var_name[]

        getsource(::Type{T}) where T <: Union{$(node_types)...} = $global_flowing_source_var_name[]

        setsource!(::T, s::U) where T <: Union{$(node_types...)} where U <: Union{$(flow_source_types...)} = begin
            $global_flowing_source_var_name[] = s
            nothing
        end

        setsource!(::Type{T}, s::U) where T <: Union{$(node_types...)} where U <: Union{$(flow_source_types...)} = begin
            $global_flowing_source_var_name[] = s
            nothing
        end

        setsource!(::T, ::Type{U}) where T <: Union{$(node_types...)} where U <: Union{$(flow_source_types...)} = begin
            $global_flowing_source_var_name[] = U()
            nothing
        end 

        setsource!(::Type{T}, ::Type{U}) where T <: Union{$(node_types...)} where U <: Union{$(flow_source_types...)} = begin
            $global_flowing_source_var_name[] = U()
            nothing
        end

    end |> eval
end

generate_lookups(::Type{T}) where T <: Tuple = begin
    for node_type in T |> fieldtypes
        str_node_type = node_type |> node_type_to_str
        value_var_name = Symbol("__v_", str_node_type)
        state_var_name = Symbol("__s_", str_node_type)
        init_val_var_name = Symbol("__iv_", str_node_type)
        init_state_var_name = Symbol("__is_", str_node_type)
        
        quote
            const $value_var_name = Ref{$(node_type |> valtype)}($(node_type |> valtype |> default))

            const $state_var_name = Ref{$(node_type |> statetype)}($(node_type |> statetype |> default))

            const $init_val_var_name = Ref{Bool}(false)

            const $init_state_var_name = Ref{Bool}(false)

            is_val_init(::$node_type) = $init_val_var_name[]

            is_val_init(::Type{$node_type}) = $init_val_var_name[]

            is_state_init(::$node_type) = $init_state_var_name[]

            is_state_init(::Type{$node_type}) = $init_state_var_name[]

            getval(::$node_type) = is_val_init($node_type) ? $value_var_name[] : error("can't access unintialized value")

            getval(::Type{$node_type}) = is_val_init($node_type) ? $value_var_name[] : error("can't access unintialized value")

            getval(::$node_type, default::$(node_type |> valtype)) = is_val_init($node_type) ? $value_var_name[] : default

            getval(::Type{$node_type}, default::$(node_type |> valtype)) = is_val_init($node_type) ? $value_var_name[] : default

            setval!(::$node_type, new_val::$(node_type |> valtype)) = begin
                $init_val_var_name[] = true
                $value_var_name[] = new_val
                nothing
            end

            setval!(::Type{$node_type}, new_val::$(node_type |> valtype)) = begin
                $init_val_var_name[] = true
                $value_var_name[] = new_val
                nothing
            end

            getstate(::$node_type) = is_state_init($node_type) ? $state_var_name[] : error("can't access unintialized state")

            getstate(::Type{$node_type}) = is_state_init($node_type) ? $state_var_name[] : error("can't access unintialized state")

            getstate(::$node_type, default::$(node_type |> statetype)) = is_state_init($node_type) ? $state_var_name[] : default

            getstate(::Type{$node_type}, default::$(node_type |> statetype)) = is_state_init($node_type) ? $state_var_name[] : default

            setstate!(::$node_type, new_state::$(node_type |> statetype)) = begin
                $init_state_var_name[] = true
                $state_var_name[] = new_state
                nothing
            end

            setstate!(::Type{$node_type}, new_state::$(node_type |> statetype)) = begin
                $init_state_var_name[] = true
                $state_var_name[] = new_state
                nothing
            end

        end |> eval
    end
end

unroll_computation(::Type{T}) where T <: Tuple = begin
    v = Vector{Expr}()
    for node_type in (T |> fieldtypes)[2:end]
        expr = quote
            node = $node_type()
            calc(node)
        end
        push!(v, expr)
    end
    Expr(:block, v...)
end

generate_flow(::Type{T}, ::Type{U}) where T <: FlowSource where U <: Tuple = begin
    quote 
        flow!(source::$T, val::$(T |> valtype)) = begin
            setsource!(source, source)
            setval!(source, val)
            $(U |> unroll_computation)
            nothing
        end

        flow!(source::Type{$T}, val::$(T |> valtype)) = begin
            setsource!(source, source)
            setval!(source, val)
            $(U |> unroll_computation)
            nothing
        end
    end |> eval
end

generate_flow(::Type{T}, ::Type{U}) where T <: NoFlowSource where U <: Tuple = begin
    quote 
        flow!(source::$T, _) = begin
            error("Cannot flow into a NoFlowSource: ", source)
        end

        flow!(source::Type{$T}, _) = begin
            error("Cannot flow into a NoFlowSource: ", source)
        end
    end |> eval
end

generate_code(comp_desc::Dict{DataType, DataType}) = begin
    comp_desc |> generate_get_source
    for (source_type, tuple_type) in comp_desc
        tuple_type |> generate_lookups
        generate_flow(source_type, tuple_type)
    end
    nothing
end

materialize(nodes) = begin
    nodes = map(nodes) do n
        if n isa Node
            n
        elseif n <: Node
            n()
        else
            error("invalid argument: ", n)
        end
    end
    comp_desc = nodes |> typeof |> expand_and_order_nodes
    comp_desc |> generate_code
    nothing
end

end