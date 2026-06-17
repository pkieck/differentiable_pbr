"""
A focused CHEMKIN-II reader for Mechanism (kinetics.jl).

Scopes: ELEMENTS, SPECIES, THERMO, REACTIONS sections.
Modified-Arrhenius reactions, integer/fractional stoichiometry.
FORD auxiliary lines, NASA-7 THERMO cards for dH and molar masses.

Not handled: reverse rates, third bodies, fall-off.
Reversible reactions print a warning; use => for explicit forward-only.
"""

@isdefined(GlobalReaction) || include("kinetics.jl")

# Standard atomic weights [kg/mol] for molar masses from element composition.
const ATOMIC_WEIGHT = Dict(
    "H" => 1.008e-3, "C" => 12.011e-3, "O" => 15.999e-3, "N" => 14.007e-3,
    "AR" => 39.948e-3, "HE" => 4.0026e-3, "S" => 32.06e-3, "CL" => 35.45e-3,
)

const _CK_TREF = 298.15   # reference T for dH [K]

# Small parsing helpers──────────

# strip a trailing `! comment`, keep the code part
_decomment(line) = first(split(line, '!'; limit=2))

# fixed-column slice (1-indexed, inclusive), tolerant of short lines
function _cols(line::AbstractString, a::Int, b::Int)
    b = min(b, lastindex(line))
    a > b && return ""
    strip(line[a:b])
end
_colf(line, a, b) = (s = _cols(line, a, b); isempty(s) ? 0.0 : parse(Float64, s))

# THERMO (NASA-7)────────────

# Parse one 4-line NASA-7 card into (name, Nasa7).  Fixed-column format per the
# CHEMKIN-II spec; see kinetics.jl `Nasa7` for how the coefficients are used.
function _parse_nasa_card(l1, l2, l3, l4, Tdefaults)
    name = first(split(l1))
    # element composition: up to four (symbol, count) pairs in cols 25–44
    Mw = 0.0
    for (ca, cb, na, nb) in ((25,26,27,29), (30,31,32,34), (35,36,37,39), (40,41,42,44))
        sym = uppercase(_cols(l1, ca, cb))
        isempty(sym) && continue
        cnt = _colf(l1, na, nb)
        cnt == 0 && continue
        haskey(ATOMIC_WEIGHT, sym) || error("unknown element '$sym' in THERMO card for $name")
        Mw += cnt * ATOMIC_WEIGHT[sym]
    end
    Tlow  = _colf(l1, 46, 55); Tlow  == 0 && (Tlow  = Tdefaults[1])
    Thigh = _colf(l1, 56, 65); Thigh == 0 && (Thigh = Tdefaults[3])
    Tmid  = _colf(l1, 66, 73); Tmid  == 0 && (Tmid  = Tdefaults[2])

    high = (_colf(l2,1,15), _colf(l2,16,30), _colf(l2,31,45), _colf(l2,46,60), _colf(l2,61,75),
            _colf(l3,1,15), _colf(l3,16,30))
    low  = (_colf(l3,31,45), _colf(l3,46,60), _colf(l3,61,75),
            _colf(l4,1,15), _colf(l4,16,30), _colf(l4,31,45), _colf(l4,46,60))
    name, Nasa7(Tlow, Tmid, Thigh, low, high, Mw)
end

function _parse_thermo(lines)
    thermo = Dict{String,Nasa7{Float64}}()
    # optional global default temperature line right after THERMO / THERMO ALL
    Tdefaults = (300.0, 1000.0, 5000.0)
    i = 1
    if i <= length(lines)
        toks = split(_decomment(lines[i]))
        if length(toks) == 3 && all(t -> tryparse(Float64, t) !== nothing, toks)
            Tdefaults = (parse(Float64, toks[1]), parse(Float64, toks[2]), parse(Float64, toks[3]))
            i += 1
        end
    end
    # cards: a line ending in column-80 marker '1' starts a 4-line block
    while i + 3 <= length(lines)
        l1 = lines[i]
        if endswith(rstrip(_decomment(l1)), "1")
            name, n = _parse_nasa_card(l1, lines[i+1], lines[i+2], lines[i+3], Tdefaults)
            thermo[uppercase(name)] = n
            i += 4
        else
            i += 1
        end
    end
    thermo
end

# ── REACTIONS ───────────────────────────────────────────────────────────────

# energy-unit → factor to J/mol, plus the quantity (concentration) unit
function _reaction_units(header_tokens)
    efac = 4.184                 # CAL/MOLE default
    e_is_kelvin = false
    moles = true                 # cm-mol-s default (vs SI)
    for t in header_tokens
        u = uppercase(t)
        u == "CAL/MOLE"     && (efac = 4.184)
        u == "KCAL/MOLE"    && (efac = 4184.0)
        u == "JOULES/MOLE"  && (efac = 1.0)
        u == "KJOULES/MOLE" && (efac = 1000.0)
        u == "KELVINS"      && (e_is_kelvin = true)
        u == "MOLES"        && (moles = true)
        (u == "SI" || u == "MKS") && (moles = false)
        u == "MOLECULES" && error("MOLECULES unit not supported by this reader")
    end
    (efac, e_is_kelvin, moles)
end

# split a reaction equation half ("CO + 3 H2") into name => coefficient
function _parse_side!(acc::Dict{String,Float64}, side)
    for term in split(side, '+')
        t = strip(term)
        (isempty(t) || uppercase(t) == "M" || uppercase(t) == "(+M)") && continue
        m = match(r"^([0-9]*\.?[0-9]+)?\s*([A-Za-z][\w\-\(\),\*]*)$", t)
        m === nothing && error("cannot parse reaction term '$t'")
        coef = m.captures[1] === nothing ? 1.0 : parse(Float64, m.captures[1])
        sp = m.captures[2]
        acc[sp] = get(acc, sp, 0.0) + coef
    end
end

# parse "FORD / CO 1.0 /" → ("CO", 1.0)
function _parse_ford(line)
    inner = match(r"/(.*)/", line)
    inner === nothing && error("malformed FORD line: $line")
    toks = split(strip(inner.captures[1]))
    length(toks) == 2 || error("malformed FORD line: $line")
    (toks[1], parse(Float64, toks[2]))
end

"""
    read_chemkin(path; Tref=298.15) → Mechanism

Parse a CHEMKIN-II `.inp` file into a `Mechanism`.  Molar masses and per-reaction
heats of reaction ΔH(Tref) come from the THERMO section (required).
"""
function read_chemkin(path::AbstractString; Tref::Float64=_CK_TREF)
    raw = readlines(path)
    # section bookkeeping
    sections = Dict{String,Vector{String}}()
    react_header = String[]
    cur = nothing
    for line in raw
        code = strip(_decomment(line))
        isempty(code) && continue
        kw = uppercase(first(split(code)))
        if kw in ("ELEMENTS", "ELEM")
            cur = "ELEMENTS"; sections[cur] = String[]; continue
        elseif kw in ("SPECIES", "SPEC")
            cur = "SPECIES"; sections[cur] = String[]; continue
        elseif kw == "THERMO"
            cur = "THERMO"; sections[cur] = String[]; continue   # keep original spacing
        elseif kw in ("REACTIONS", "REAC")
            cur = "REACTIONS"; sections[cur] = String[]
            react_header = split(code)[2:end]; continue
        elseif kw == "END"
            cur = nothing; continue
        end
        cur === nothing && continue
        # THERMO needs verbatim column spacing; everything else can be de-commented
        push!(sections[cur], cur == "THERMO" ? rstrip(line) : code)
    end

    haskey(sections, "SPECIES")  || error("CHEMKIN file has no SPECIES section")
    haskey(sections, "THERMO")   || error("this reader requires a THERMO section (for ΔH & molar masses)")
    haskey(sections, "REACTIONS")|| error("CHEMKIN file has no REACTIONS section")

    species = String[]
    for l in sections["SPECIES"]; append!(species, split(l)); end
    Ns = length(species)
    upspec = uppercase.(species)
    sidx(name) = (idx = findfirst(==(uppercase(name)), upspec);
                  idx === nothing ? error("species '$name' used in a reaction but not in SPECIES") : idx)

    thermo = _parse_thermo(sections["THERMO"])
    M = Float64[(haskey(thermo, s) ? thermo[s].Mw :
                 error("species '$s' missing from THERMO")) for s in upspec]

    efac, e_is_kelvin, moles = _reaction_units(react_header)

    # group reaction lines with their following FORD/aux lines
    reactions = GlobalReaction{Float64,Ns}[]
    rlines = sections["REACTIONS"]
    i = 1
    while i <= length(rlines)
        line = rlines[i]
        if !occursin('=', line)         # stray aux without a reaction → skip
            i += 1; continue
        end
        # reversibility
        reversible = !occursin("=>", line) || occursin("<=>", line)
        arrow = occursin("<=>", line) ? "<=>" : occursin("=>", line) ? "=>" : "="
        eqpart, _, rest = partition_reaction(line, arrow)
        toks = split(rest)
        length(toks) >= 3 || error("reaction missing Arrhenius A β Ea: $line")
        A_in, beta, Ea_in = parse.(Float64, toks[end-2:end])

        lhs, rhs = split(eqpart, arrow; limit=2)
        react = Dict{String,Float64}(); prod = Dict{String,Float64}()
        _parse_side!(react, lhs); _parse_side!(prod, rhs)

        # net stoichiometry & default forward orders (= reactant coefficients)
        nu    = zeros(Float64, Ns)
        order = zeros(Float64, Ns)
        for (sp, c) in react; nu[sidx(sp)] -= c; order[sidx(sp)] += c; end
        for (sp, c) in prod;  nu[sidx(sp)] += c; end

        # consume following FORD aux lines (and ignore DUP/REV/LOW/TROE)
        j = i + 1
        while j <= length(rlines) && !occursin('=', rlines[j])
            aux = uppercase(first(split(rlines[j])))
            if aux == "FORD"
                sp, val = _parse_ford(rlines[j]); order[sidx(sp)] = val
            elseif aux in ("REV", "LOW", "TROE", "PLOG", "RORD", "DUP", "DUPLICATE")
                # not modelled by this reader; silently skip
            end
            j += 1
        end
        i = j

        reversible && @warn "reaction '$eqpart' is reversible; only the forward rate is built (author as `=>` to silence)."

        Ea = e_is_kelvin ? Ea_in * R_GAS : Ea_in * efac
        n_overall = sum(order)
        A = moles ? A_in * (1e6)^(1 - n_overall) : A_in    # cm-mol-s → SI

        # heat of reaction at Tref from NASA enthalpies
        dH = 0.0
        for k in 1:Ns
            nu[k] == 0 && continue
            dH += nu[k] * enthalpy(thermo[upspec[k]], Tref)
        end

        push!(reactions, GlobalReaction(A, beta, Ea,
              ntuple(k -> nu[k],    Ns), ntuple(k -> order[k], Ns),
              dH, ntuple(k -> M[k], Ns)))
    end

    Mechanism(species, M, Tuple(reactions))
end

# split "eq  arrow  A b Ea" returning (equation_with_arrow, arrow, trailing)
function partition_reaction(line, arrow)
    # the equation is everything up to and including the LAST token before the
    # three Arrhenius numbers; easiest is to locate the arrow, then peel the
    # trailing 3 numeric tokens off the remainder.
    a = findfirst(arrow, line)
    before = line[1:prevind(line, first(a))]
    after  = line[nextind(line, last(a)):end]
    toks = split(after)
    nnum = length(toks)
    # products = all but last 3 tokens
    prod_str = join(toks[1:nnum-3], ' ')
    eq = strip(before) * " " * arrow * " " * prod_str
    (strip(eq), arrow, join(toks[nnum-2:nnum], ' '))
end
