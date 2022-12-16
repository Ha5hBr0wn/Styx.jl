module Styx

################# using statements ################
using DataStructures
using Base: Base.Order.Ordering, Base.Order.Forward, Base.Order.ForwardOrdering



################# node defs ################
abstract type Node{V, S} end


abstract type AbstractSource{ID, V, S} <: Node{V, S} end


abstract type AbstractComputation{V, S} <: Node{V, S} end


struct FlowSource{ID, V} <: AbstractSource{ID, V, Missing} end


struct NoFlowSource{ID, V} <: AbstractSource{ID, V, Missing} end


struct NoLink{T <: Node, V, S} <: AbstractComputation{V, S} end


struct AsyncGroupStart <: Node{Missing, Missing} end


struct AsyncGroupEnd <: Node{Missing, Missing} end


struct Async{T <: Node, V, S} <: AbstractComputation{V, S} end


struct FieldSplitter{T <: Node, I, V} <: AbstractComputation{V, Missing} end


struct Combine{T <: Tuple, V} <: AbstractComputation{V, Missing} end


struct CumulativeSum{T <: Node, V} <: AbstractComputation{V, Missing} end


struct CumulativeProduct{T <: Node, V} <: AbstractComputation{V, Missing} end


struct Sum{T <: Tuple, V} <: AbstractComputation{V, Missing} end


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


struct Pow{T <: Node{<:Real}, P} <: AbstractComputation{Float64, Missing} end



################# node constructors ################
const source_id_set = Set{Int64}(0)

FlowSource(id::Int64, value_type::DataType) = FlowSource{id, value_type}()

FlowSource(value_type::DataType) = begin
    r = 0
    while r in source_id_set
        r = rand(Int64)
    end
    push!(source_id_set, r)
    FlowSource(r, value_type)
end


NoFlowSource(id::Int64, value_type::DataType) = NoFlowSource{id, value_type}()

NoFlowSource(value_type::DataType) = begin
    r = 0
    while r in source_id_set
        r = rand(Int64)
    end
    push!(source_id_set, r)
    NoFlowSource(r, value_type)
end


NoLink(n::Node) = NoLink{n |> typeof, n |> valtype, n |> statetype}()


Async(n::Node) = Async{n |> typeof, n |> valtype, n |> statetype}()


FieldSplitter(n::Node, i::Int64) = FieldSplitter{n |> typeof, i, (n |> valtype |> fieldtypes)[i]}()


Split(n::Node) = begin
    output_type = n |> valtype
    num_fields = output_type |> fieldcount

    field_splitters = Vector{FieldSplitter}()
    for i in 1:num_fields
        push!(field_splitters, FieldSplitter(n, i))
    end

    tuple(field_splitters...)
end

Split(n::Node, symbols::Vararg) = begin
    output_type = valtype(n)
    field_names = fieldnames(output_type)
    symbol_idxs = Vector{Int64}()

    # Find symbol idxs in order of symbols
    for symbol in symbols
        push!(symbol_idxs, findfirst(x -> x == symbol, field_names))
    end

    # Create field splitters 
    field_splitters = Vector{FieldSplitter}()
    for idx in symbol_idxs
        push!(field_splitters, FieldSplitter(n, idx))
    end

    tuple(field_splitters...)
end


Combine(value_type::DataType, ns::Vararg) = Combine{ns |> typeof, value_type}()


CumulativeSum(n::Node) = CumulativeSum{n |> typeof, n |> valtype}()


CumulativeProduct(n::Node) = CumulativeProduct{n |> typeof, n |> valtype}()


Sum(ns::Vararg) = Sum{ns |> typeof, promote_type(map(valtype, ns)...)}()

Sum(output_type::DataType, ns::Vararg) = Sum{ns |> typeof, output_type}()


Difference(lhs::Node, rhs::Node) = Difference{lhs |> typeof, rhs |> typeof, promote_type(lhs |> valtype, rhs |> valtype)}()


Product(lhs::Node, rhs::Node) = Product{lhs |> typeof, rhs |> typeof, promote_type(lhs |> valtype, rhs |> valtype)}()


Quotient(lhs::Node, rhs::Node) = Quotient{lhs |> typeof, rhs |> typeof}()


Previous(n::Node) = Previous{n |> typeof, n |> valtype}()


Delta(n::Node) = Difference(n, Previous(n))


Buffer(n::Node, size::Int64) = Buffer{n |> typeof, size, n |> valtype}()


Collector(n::Node) = Collector{n |> typeof, n |> valtype}()


ConditionalCollector(n::Node, condition::Function) = ConditionalCollector{n |> typeof, condition |> typeof, n |> valtype}()


SimpleMA(n::Node, size::Int64) = SimpleMA{Buffer(n, size) |> typeof, n |> valtype}()


SimpleEMA(n::Node, alpha::Float64) = SimpleEMA{n |> typeof, alpha}()


Maximum(n::Node, order::Ordering) = Maximum{n |> typeof, order |> typeof, n |> valtype}()

Maximum(n::Node) = Maximum(n, Forward)


Minimum(n::Node, order::Ordering) = Minimum{n |> typeof, order |> typeof, n |> valtype}()

Minimum(n::Node) = Minimum(n, Forward)


MaxDrawDown(n::Node) = Maximum(
    Difference(
        Maximum(n), 
        n
    )
)


Negative(n::Node) = Negative{n |> typeof, n |> valtype}()


Abs(n::Node) = Abs{n |> typeof, n |> valtype}()


Counter(n::Node) = Counter{n |> typeof}()


Mean(n::Node) = CumulativeSum(n) / Counter(n)


Square(n::Node) = Product(n, n)


Variance(n::Node) = Mean(Square(n)) - Square(Mean(n))


Pow(n::Node, p::Real) = Pow{n |> typeof, p}()


StdDev(n::Node) = Pow(
    Variance(n), 
    0.5
)



################ node calcs ################
@inline calc(n::NoLink{T}) where T = begin
    if is_val_init(T)
        setval!(n, T |> getval)
    end
    if is_state_init(T)
        setstate!(n, T |> getstate)
    end
    nothing
end

@inline calc(n::Async{T}) where T = begin
    calc(T())
    if is_val_init(T)
        setval!(n, T |> getval)
    end
    if is_state_init(T)
        setstate!(n, T |> getstate)
    end
    nothing
end

@inline calc(n::FieldSplitter{T, I}) where {T, I} = begin
    if is_val_init(T)
        val = T |> getval
        setval!(n, getfield(val, I))
    end
    nothing
end

@inline calc(n::Combine{T, V}) where {T, V} = begin
    # Set value by calling the output type constructor with all dependency values
    if all_vals_init(T)
        vals = getvals(T)
        setval!(n, V(vals...))
    end

    nothing
end

@inline calc(n::Combine{T, V}) where {T, V <: Tuple} = begin
    if all_vals_init(T)
        setval!(n, T |> getvals)
    end

    nothing
end

@inline calc(n::CumulativeSum{T}) where T = begin
    if is_val_init(T)
        setval!(n, getval(n, zero(n |> valtype)) + getval(T))
    end
    nothing
end

@inline calc(n::CumulativeProduct{T}) where T = begin
    if is_val_init(T)
        setval!(n, getval(n, one(n |> valtype)) * getval(T))
    end
    nothing
end

@inline calc(n::Sum{T}) where T = begin
    if all_vals_init(T)
        setval!(n, sum(T |> getvals))
    end
    nothing
end

@inline calc(n::Difference{LHS, RHS}) where {LHS, RHS} = begin
    if is_val_init(LHS) && is_val_init(RHS)
        setval!(n, getval(LHS) - getval(RHS))
    end
    nothing
end

@inline calc(n::Product{LHS, RHS}) where {LHS, RHS} = begin
    if is_val_init(LHS) && is_val_init(RHS)
        setval!(n, getval(LHS) * getval(RHS))
    end
    nothing
end

@inline calc(n::Quotient{LHS, RHS}) where {LHS, RHS} = begin
    if is_val_init(LHS) && is_val_init(RHS)
        setval!(n, getval(LHS) / getval(RHS))
    end
    nothing
end

@inline calc(n::Previous{T}) where T = begin
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

@inline calc(n::Buffer{T, Size}) where {T, Size} = begin
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

@inline calc(n::Collector{T}) where T = begin
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

@inline calc(n::ConditionalCollector{T, U}) where {T, U} = begin
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

@inline calc(n::SimpleMA{B}) where B <: Buffer{T, Size} where {T, Size} = begin
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

@inline calc(n::SimpleEMA{T, Alpha}) where {T, Alpha} = begin
    if is_val_init(T)
        ema = getval(n, 0.0)
        next_val = T |> getval
        ema = (1 - Alpha)ema + (Alpha)next_val
        setval!(n, ema)
    end
    nothing
end

@inline calc(n::Maximum{T, O}) where {T, O} = begin
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

@inline calc(n::Minimum{T, O}) where {T, O} = begin
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

@inline calc(n::Negative{T}) where T = begin
    if is_val_init(T)
        val = T |> getval
        setval!(n, -val)
    end
    nothing
end

@inline calc(n::Abs{T}) where T = begin
    if is_val_init(T)
        setval!(n, T |> getval |> abs)
    end
    nothing
end

@inline calc(n::Counter{T}) where T = begin
    if !is_val_init(n) && is_val_init(T)
        setval!(n, 1)
    elseif is_val_init(T)
        setval!(n, (n |> getval) + 1)
    end
    nothing
end

@inline calc(n::Pow{T, P}) where {T, P} = begin
    if is_val_init(T)
        val = T |> getval
        pow = convert(Float64, val^P)
        setval!(n, pow)
    end
    nothing
end



################### calc helpers ##################
@generated all_vals_init(::Type{T}) where T <: Tuple = begin
    v = Vector{Expr}()
    
    for node_type in fieldtypes(T)
        e = quote
            if !is_val_init($node_type) return false end
        end

        push!(v, e)
    end

    push!(v, :(return true))

    Expr(:block, v...)
end


@generated all_states_init(::Type{T}) where T <: Tuple = begin
    v = Vector{Expr}()
    
    for node_type in fieldtypes(T)
        e = quote
            if !is_state_init($node_type) return false end
        end

        push!(v, e)
    end

    push!(v, :(return true))

    Expr(:block, v...)
end


@generated getvals(::Type{T}) where T <: Tuple = begin
    v = Vector{Expr}()

    for node_type in fieldtypes(T)
        push!(v, :($node_type |> getval))
    end

    Expr(:tuple, v...)
end


@generated getstates(::Type{T}) where T <: Tuple = begin
    v = Vector{Expr}()

    for node_type in fieldtypes(T)
        push!(v, :($node_type |> getstate))
    end

    Expr(:tuple, v...)
end



############### node type introspection valtype/statetype #################
@inline valtype(::Type{T}) where T <: Node{V, S} where {V, S} = V

@inline valtype(::Node{V, S}) where {V, S} = V


@inline statetype(::Type{T}) where T <: Node{V, S} where {V, S} = S

@inline statetype(::Node{V, S}) where {V, S} = S


@inline synctype(::Type{T}) where T <: Async{U} where U = U

@inline synctype(::Async{T}) where T = T


@inline sourceid(::Type{T}) where T <: AbstractSource{ID} where ID = ID

@inline sourceid(::AbstractSource{ID}) where ID = ID



############## empty definitions of metaprogramming defined functions #################
@inline getsource(_) = error("Undefined method")


@inline setsource!(_, _) = error("Undefined method")


@inline getval(_) = error("Undefined method")

@inline getval(_, _) = error("Undefined method")


@inline setval!(_, _) = error("Undefined method")


@inline getstate(_) = error("Undefined method")

@inline getstate(_, _) = error("Undefined method")


@inline setstate!(_, _) = error("Undefined method")


@inline is_val_init(_) = error("Undefined method")


@inline is_state_init(_) = error("Undefined method")


@inline flow!(_, _) = error("Undefined method")



##################### expanding and ordering nodes, manipulating the type based DAG #######################
build_dag(node_types) = begin
    # Get all types explicitly
    expanded_types = node_types |> get_expanded_types

    # Create graph structure and necessary metdata
    in_graph, out_graph, flow_sources, no_flow_sources, async_nodes = expanded_types |> init_type_graph

    # Edit the graph structure to account for async_nodes (remove the sync_nodes and replace with the async version)
    reconnect_async_nodes!(in_graph, out_graph, async_nodes)

    # Get the computational descriptor (a map from a flow_source to the sequence of nodes to execute)
    comp_desc = get_computational_descriptor(in_graph, out_graph, flow_sources)

    # Get the disconnected nodes that still need code generation (sync_nodes and no_flow_sources)
    disconnected_nodes = get_disconnected_nodes(no_flow_sources, async_nodes)

    # return 
    comp_desc, disconnected_nodes
end


get_expanded_types(node_types) = begin
    s = Set{DataType}()

    for node_type in node_types
        union!(s, node_type |> expand_type)
    end

    s
end


expand_type(type::DataType) = begin
    s = Set{DataType}()
    
    if type <: Node
        push!(s, type)
        for type_param in type.parameters
            union!(s, type_param |> expand_type)
        end
    elseif type <: Tuple
        for field_type in fieldtypes(type)
            union!(s, field_type |> expand_type)
        end
    else 
        nothing
    end

    s
end

expand_type(_) = Set{DataType}() # Handles value type parameters


init_type_graph(node_types::Set{DataType}) = begin
    # Allocate outputs
    in_graph = Dict(t => Set{DataType}() for t in node_types)
    out_graph = Dict(t => Set{DataType}() for t in node_types)
    flow_sources = Set{DataType}()
    no_flow_sources = Set{DataType}()
    async_nodes = Set{DataType}()

    for node_type in node_types
        # Push node_type to metadata if needed
        if node_type <: FlowSource 
            push!(flow_sources, node_type) 
        elseif node_type <: NoFlowSource
            push!(no_flow_sources, node_type)
        elseif node_type <: Async
            push!(async_nodes, node_type)
        end

        # Create appropriate edges in in_graph and out_graph
        for type_param in node_type.parameters
            connect_edges!(in_graph, out_graph, node_type, type_param)
        end
    end

    in_graph, out_graph, flow_sources, no_flow_sources, async_nodes
end


connect_edges!(in_graph::Dict, out_graph::Dict, node_type::DataType, type_param) = begin
    if type_param isa DataType && (type_param <: NoLink || type_param <: NoFlowSource)
        nothing
    elseif type_param isa DataType && type_param <: Node
        push!(out_graph[type_param], node_type)
        push!(in_graph[node_type], type_param)
    elseif type_param isa DataType && type_param <: Tuple
        for field_type in fieldtypes(type_param)
            connect_edges!(in_graph, out_graph, node_type, field_type)
        end
    end

    nothing
end


reconnect_async_nodes!(in_graph::Dict, out_graph::Dict, async_nodes::Set) = begin
    for async_node in async_nodes
        sync_node = async_node |> synctype
        
        length(out_graph[sync_node]) == 1 || error("node connected to synchronous version of an async node")
        pop!(out_graph[sync_node])

        in_graph[async_node] = in_graph[sync_node]
        in_graph[sync_node] = Set{DataType}()

        for node in in_graph[async_node]
            pop!(out_graph[node], sync_node)
            push!(out_graph[node], async_node)
        end
    end
end


get_computational_descriptor(in_graph::Dict, out_graph::Dict, flow_sources::Set) = begin
    comp_desc = Dict{DataType, Vector{DataType}}()

    for flow_source in flow_sources
        in_graph_prime, out_graph_prime = get_reachable_component(flow_source, in_graph, out_graph)
        push!(comp_desc, flow_source => topological_sort(flow_source, in_graph_prime, out_graph_prime))
    end

    comp_desc
end


get_reachable_component(u::DataType, in_graph::Dict, out_graph::Dict) = begin
    # Setup
    stack = [u]
    in_graph_prime = Dict{DataType, Set{DataType}}()
    out_graph_prime = Dict{DataType, Set{DataType}}()
    visited = Set{DataType}()
    
    # Run DFS
    while !isempty(stack)
        v = pop!(stack)
        
        if !(v in visited)
            push!(visited, v)            
            append!(stack, out_graph[v])
        end
    end

    # Construct reachable component 
    for v in visited
        in_graph_prime[v] = intersect(in_graph[v], visited)
        out_graph_prime[v] = out_graph[v]
    end
    
    return in_graph_prime, out_graph_prime
end


topological_sort(source::DataType, in_graph::Dict, out_graph::Dict) = begin    
    # Setup
    stack = Vector{DataType}()
    async_group = Vector{DataType}()
    volatile_in_graph = deepcopy(in_graph)
    top_sort = Vector{DataType}()

    # Inititalize stack
    push!(stack, source)

    # Run algorithm
    while !isempty(stack) || !isempty(async_group)
        if isempty(stack)
            new_async_group = Vector{DataType}()

            push!(top_sort, AsyncGroupStart)

            while !isempty(async_group)
                update_top_sort!(top_sort, async_group, new_async_group, stack, volatile_in_graph, out_graph)
            end

            push!(top_sort, AsyncGroupEnd)

            async_group = new_async_group
        else
            update_top_sort!(top_sort, stack, async_group, stack, volatile_in_graph, out_graph)
        end
    end

    top_sort
end


update_top_sort!(top_sort::Vector, pop_stack::Vector, push_async_stack::Vector, push_stack::Vector, in_graph::Dict, out_graph::Dict) = begin
    u = pop!(pop_stack)
    push!(top_sort, u)
    
    for v in out_graph[u]
        pop!(in_graph[v], u)
        if isempty(in_graph[v])
            if v <: Async
                push!(push_async_stack, v)
            else
                push!(push_stack, v)
            end
        end
    end

    nothing
end


get_disconnected_nodes(no_flow_sources::Set, async_nodes::Set) = begin
    sync_nodes = Set{DataType}()
    for async_node in async_nodes
        push!(sync_nodes, async_node |> synctype)
    end

    union(no_flow_sources, sync_nodes)
end



##################### implicitly generate required methods ########################
generate_getset_node(node_type::DataType) = begin
    val_var_name = gensym()
    state_var_name = gensym()
    init_val_var_name = gensym()
    init_state_var_name = gensym()

    backup_val_var_name = gensym()
    backup_state_var_name = gensym()
    backup_init_val_var_name = gensym()
    backup_init_state_var_name = gensym()
    
    quote
        # Define variables
        const $val_var_name = Ref{$(node_type |> valtype)}()

        const $state_var_name = Ref{$(node_type |> statetype)}()

        const $init_val_var_name = Ref{Bool}(false)

        const $init_state_var_name = Ref{Bool}(false)                    
            
        
        # Check if initialized
        @inline is_val_init(::$node_type) = $init_val_var_name[]

        @inline is_val_init(::Type{$node_type}) = $init_val_var_name[]

        @inline is_state_init(::$node_type) = $init_state_var_name[]

        @inline is_state_init(::Type{$node_type}) = $init_state_var_name[]

        
        # Get and set val
        @inline getval(::Type{$node_type}) = $init_val_var_name[] ? $val_var_name[] : error("can't access unintialized val")

        @inline getval(::$node_type) = getval($node_type)

        @inline getval(::Type{$node_type}, default::$(node_type |> valtype)) = is_val_init($node_type) ? $val_var_name[] : default

        @inline getval(::$node_type, default::$(node_type |> valtype)) = getval($node_type, default)

        @inline setval!(::Type{$node_type}, new_val::$(node_type |> valtype)) = begin
            $init_val_var_name[] = true
            $val_var_name[] = new_val
            nothing
        end

        @inline setval!(::$node_type, new_val::$(node_type |> valtype)) = setval!($node_type, new_val)


        # Get and set state
        @inline getstate(::Type{$node_type}) = $init_state_var_name[] ? $state_var_name[] : error("can't access unintialized state")

        @inline getstate(::$node_type) = getstate($node_type)

        @inline getstate(::Type{$node_type}, default::$(node_type |> statetype)) = $init_state_var_name[] ? $state_var_name[] : default

        @inline getstate(::$node_type, default::$(node_type |> statetype)) = getstate($node_type, default)

        @inline setstate!(::Type{$node_type}, new_state::$(node_type |> statetype)) = begin
            $init_state_var_name[] = true
            $state_var_name[] = new_state
            nothing
        end

        @inline setstate!(::$node_type, new_state::$(node_type |> statetype)) = setstate!($node_type, new_state)

        
        # Backup and restore
        const $backup_val_var_name = Ref{$(node_type |> valtype)}()
        
        const $backup_state_var_name = Ref{$(node_type |> statetype)}()
        
        const $backup_init_val_var_name = Ref{Bool}(false)
        
        const $backup_init_state_var_name = Ref{Bool}(false)

        @inline backup!(::Type{$node_type}) = begin
            $backup_val_var_name[] = $val_var_name[]
            $backup_state_var_name[] = $state_var_name[]
            $backup_init_val_var_name[] = $init_val_var_name[]
            $backup_init_state_var_name[] = $init_state_var_name[]
        end

        @inline backup!(::$node_type) = backup!($node_type)

        @inline restore!(::Type{$node_type}) = begin
            $val_var_name[] = $backup_val_var_name[]
            $state_var_name[] = $backup_state_var_name[]
            $init_val_var_name[] = $backup_init_val_var_name[]
            $init_state_var_name[] = $backup_init_state_var_name[]
        end

        @inline restore!(::$node_type) = restore!($node_type)

    end |> eval
end


wait_on_futures(future_var_names::Vector{Symbol}) = begin
    v = Vector{Expr}()

    for var_name in future_var_names
        expr = quote
            wait($var_name)
        end
        push!(v, expr)
    end

    Expr(:block, v...)
end


unroll_computation(top_sort::Vector{DataType}) = begin
    v = Vector{Expr}()
    future_var_names = Vector{Symbol}()

    for node_type in top_sort[2:end]
        if node_type <: AsyncGroupStart
            expr = :()
        elseif node_type <: AsyncGroupEnd
            expr = future_var_names |> wait_on_futures
            empty!(future_var_names)
        elseif node_type <: Async
            future_var_name = gensym()
            push!(future_var_names, future_var_name)

            expr = quote
                $future_var_name = Threads.@spawn calc($node_type())
            end
        else
            expr = quote
                calc($node_type())
            end
        end

        push!(v, expr)
    end
    
    Expr(:block, v...)
end


generate_flow(source_type::DataType, top_sort::Vector; checkpoint_interval=0, use_atomic_flow=false, checkpoint_on_error=false) = begin
    if checkpoint_interval <= 0 && !use_atomic_flow
        quote 
            flow!(source::Type{$source_type}, val::$(source_type |> valtype)) = begin
                setval!(source, val)
                $(top_sort |> unroll_computation)
                nothing
            end

            @inline flow!(::$source_type, val::$(source_type |> valtype)) = flow!($source_type, val)
        end |> eval

    else if checkpoint_interval <= 0 && use_atomic_flow && !checkpoint_on_error
        quote 
            flow!(source::Type{$source_type}, val::$(source_type |> valtype)) = begin
                full_backup!(source)
                try
                    setval!(source, val)
                    $(top_sort |> unroll_computation)
                catch
                    full_restore!(source)
                    rethrow()
                end
                nothing
            end

            @inline flow!(::$source_type, val::$(source_type |> valtype)) = flow!($source_type, val)
        end |> eval

    else
        error("Unsupported configuration")
    end
end


generate_full_backup(source_type::DataType, top_sort::Vector{DataType}, disconnected_nodes::Set{DataType}) = begin
    quote
        full_backup!(source::Type{$source_type}) = nothing

        full_backup!(::$source_type) = full_backup!($source_type)
    end |> eval
end


generate_full_restore(source_type::DataType, top_sort::Vector{DataType}, disconnected_nodes::Set{DataType}) = begin
    quote
        full_restore!(source::Type{$source_type}) = nothing

        full_restore!(::$source_type) = full_restore!($source_type)
    end |> eval
end


comp_desc_to_node_type_set(comp_desc::Dict) = begin
    node_types = Set{DataType}()
    
    for node_type_vector in values(comp_desc)
        # Don't include AsyncGroupStart/End
        union!(node_types, [node_type for node_type in node_type_vector if node_type <: Union{AbstractComputation, FlowSource}])
    end
    
    node_types
end


generate_code(comp_desc::Dict, disconnected_nodes::Set; checkpoint_interval=0, use_atomic_flow=false, checkpoint_on_error=false) = begin
    node_types = comp_desc |> comp_desc_to_node_type_set

    for node_type in union(node_types, disconnected_nodes)
        generate_getset_node(node_type)
    end

    for (source_type, top_sort) in comp_desc 
        if use_atomic_flow
            generate_full_backup(source_type, top_sort, disconnected_nodes)       
            generate_full_restore(source_type, top_sort, disconnected_nodes)
        end
        generate_flow(source_type, top_sort; checkpoint_interval, use_atomic_flow, checkpoint_on_error)
    end

    nothing
end


verify_atomic(node_type::DataType) = begin
    hasmethod(rev_calc, Tuple{node_type}) ||
    ((valtype(node_type) |> isbitstype) && (statetype(node_type) |> isbitstype)) ||
    error("$node_type can not be used in atomic operations")
end

verify_atomic(top_sort::Vector) = begin
    for node_type in top_sort
        verify_atomic(node_type)
    end
end

verify_atomic(comp_desc::Dict) = begin
    for top_sort in values(comp_desc)
        verify_atomic(top_sort)
    end
end


materialize(nodes; checkpoint_interval::Int64 = -1, use_atomic_flow::Bool = false, checkpoint_on_error::Bool = false) = begin
    # Check kwarg consistency
    @assert !checkpoint_on_error || use_atomic_flow
    
    # Map all nodes to their types
    node_types = map(nodes) do n
        if n isa Node || n isa Tuple
            n |> typeof
        else
            error("invalid argument: $n")
        end
    end

    # Organize type DAG
    comp_desc, disconnected_nodes = build_dag(node_types)

    # Verify atomic operations are possible 
    if use_atomic_flow
        verify_atomic(comp_desc)
    end

    # Generate all requried code
    generate_code(comp_desc, disconnected_nodes; checkpoint_interval, use_atomic_flow, checkpoint_on_error)

    nothing
end

materialize(node::Node; checkpoint_interval=0, use_atomic_flow=false, checkpoint_on_error=false) = begin
    materialize([node]; checkpoint_interval, use_atomic_flow, checkpoint_on_error)
end

end