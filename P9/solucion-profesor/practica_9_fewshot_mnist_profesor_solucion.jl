# Práctica 9 - Few-shot Learning con MNIST
# Solución del profesor (Julia)
#
# Escenario:
# 1. Entrenamos una CNN sin ver el dígito 7.
# 2. Congelamos mentalmente esa CNN y usamos la penúltima capa como extractor.
# 3. Construimos prototipos con pocos ejemplos por clase.
# 4. Clasificamos por distancia euclídea al prototipo más cercano.
# 5. Comparamos 1-shot frente a 5-shot.
#
# Paquetes necesarios:
# ] add Flux MLDatasets Statistics Random LinearAlgebra StatsBase OneHotArrays MultivariateStats Plots

using Flux
using MLDatasets
using Statistics
using Random
using LinearAlgebra
using StatsBase
using OneHotArrays
using MultivariateStats
using Plots

const SEED = 42
Random.seed!(SEED)

const NOVEL_CLASS = 7
const KNOWN_CLASSES = [0, 1, 2, 3, 4, 5, 6, 8, 9]
const COMPACT_CLASSES = collect(1:length(KNOWN_CLASSES))

function load_mnist_world_without_sevens()
    train_x, train_y = MNIST(:train)[:]
    test_x, test_y = MNIST(:test)[:]

    X_train = Float32.(reshape(train_x, 28, 28, 1, :)) ./ 255f0
    X_test = Float32.(reshape(test_x, 28, 28, 1, :)) ./ 255f0
    y_train = Int.(train_y)
    y_test = Int.(test_y)

    known_mask = y_train .!= NOVEL_CLASS
    seven_mask = y_train .== NOVEL_CLASS

    X_train_known = X_train[:, :, :, known_mask]
    y_train_known = y_train[known_mask]
    X_train_seven = X_train[:, :, :, seven_mask]
    y_train_seven = y_train[seven_mask]

    return X_train_known, y_train_known, X_train_seven, y_train_seven, X_test, y_test
end

function remap_known_labels(y)
    mapping = Dict(label => i for (i, label) in enumerate(KNOWN_CLASSES))
    return [mapping[label] for label in y]
end

function build_classifier(num_classes::Int)
    return Chain(
        Conv((3, 3), 1 => 32, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, relu),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(1600 => 64, relu),
        Dense(64 => num_classes)
    )
end

function train_classifier!(model, X_train_known, y_train_known; epochs=3, batchsize=128, lr=1f-3)
    y_compact = remap_known_labels(y_train_known)
    y_onehot = onehotbatch(y_compact, COMPACT_CLASSES)

    data = Flux.DataLoader((X_train_known, y_onehot), batchsize=batchsize, shuffle=true)
    opt_state = Flux.setup(Adam(lr), model)

    for epoch in 1:epochs
        total_loss = 0f0
        nbatches = 0

        for (xb, yb) in data
            loss, grads = Flux.withgradient(model) do m
                logits = m(xb)
                Flux.logitcrossentropy(logits, yb)
            end
            Flux.update!(opt_state, model, grads[1])
            total_loss += Float32(loss)
            nbatches += 1
        end

        println("Epoch $(epoch) - loss = ", round(total_loss / nbatches, digits=4))
    end
end

function evaluate_known_classifier(model, X_test, y_test)
    known_mask = y_test .!= NOVEL_CLASS
    X = X_test[:, :, :, known_mask]
    y_original = y_test[known_mask]
    y_compact = remap_known_labels(y_original)

    logits = model(X)
    pred_compact = onecold(logits, COMPACT_CLASSES)
    return mean(pred_compact .== y_compact)
end

function create_feature_extractor(model)
    return Chain(model.layers[1:end-1]...)
end

function sample_class_examples(X, y, class_label::Int, n::Int)
    idx = findall(==(class_label), y)
    if length(idx) < n
        error("No hay suficientes ejemplos para la clase $(class_label): se piden $(n), hay $(length(idx)).")
    end
    selected = sample(idx, n; replace=false)
    return X[:, :, :, selected], y[selected]
end

function build_fewshot_episode(X_train_known, y_train_known, X_train_seven, y_train_seven, X_test, y_test; n_shots=5, n_query_per_class=100)
    support_x_parts = Vector{Array{Float32, 4}}()
    support_y_parts = Vector{Vector{Int}}()
    query_x_parts = Vector{Array{Float32, 4}}()
    query_y_parts = Vector{Vector{Int}}()

    for class_label in 0:9
        if class_label == NOVEL_CLASS
            Xs, ys = sample_class_examples(X_train_seven, y_train_seven, class_label, n_shots)
        else
            Xs, ys = sample_class_examples(X_train_known, y_train_known, class_label, n_shots)
        end

        Xq, yq = sample_class_examples(X_test, y_test, class_label, n_query_per_class)

        push!(support_x_parts, Xs)
        push!(support_y_parts, ys)
        push!(query_x_parts, Xq)
        push!(query_y_parts, yq)
    end

    X_support = cat(support_x_parts...; dims=4)
    y_support = vcat(support_y_parts...)
    X_query = cat(query_x_parts...; dims=4)
    y_query = vcat(query_y_parts...)

    return X_support, y_support, X_query, y_query
end

function compute_prototypes(feature_extractor, X_support, y_support)
    embeddings = feature_extractor(X_support)
    # embeddings tiene forma embedding_dim x n_examples.
    class_labels = sort(unique(y_support))

    prototype_columns = Vector{Vector{Float32}}()
    for class_label in class_labels
        idx = findall(==(class_label), y_support)
        class_embeddings = embeddings[:, idx]
        prototype = vec(mean(class_embeddings; dims=2))
        push!(prototype_columns, Float32.(prototype))
    end

    prototypes = hcat(prototype_columns...)
    return prototypes, class_labels
end

function classify_by_nearest_prototype(feature_extractor, X_query, prototypes, prototype_labels)
    query_embeddings = feature_extractor(X_query)
    n_query = size(query_embeddings, 2)
    n_proto = size(prototypes, 2)

    predictions = Vector{Int}(undef, n_query)

    for i in 1:n_query
        query_vector = query_embeddings[:, i]
        distances = [norm(query_vector .- prototypes[:, j]) for j in 1:n_proto]
        nearest = argmin(distances)
        predictions[i] = prototype_labels[nearest]
    end

    return predictions
end

function confusion_matrix(y_true, y_pred; labels=collect(0:9))
    label_to_idx = Dict(label => i for (i, label) in enumerate(labels))
    cm = zeros(Int, length(labels), length(labels))
    for (true_label, pred_label) in zip(y_true, y_pred)
        i = label_to_idx[true_label]
        j = label_to_idx[pred_label]
        cm[i, j] += 1
    end
    return cm
end

function run_episode(feature_extractor, X_train_known, y_train_known, X_train_seven, y_train_seven, X_test, y_test; n_shots=5)
    X_support, y_support, X_query, y_query = build_fewshot_episode(
        X_train_known, y_train_known,
        X_train_seven, y_train_seven,
        X_test, y_test;
        n_shots=n_shots,
        n_query_per_class=100,
    )

    prototypes, prototype_labels = compute_prototypes(feature_extractor, X_support, y_support)
    y_pred = classify_by_nearest_prototype(feature_extractor, X_query, prototypes, prototype_labels)
    acc = mean(y_pred .== y_query)

    return acc, y_pred, y_query, X_query, prototypes, prototype_labels
end

function plot_accuracy_comparison(results::Dict{String, Float64}; output_path="fewshot_accuracy_comparison_julia.png")
    labels = collect(keys(results))
    values = [results[label] for label in labels]

    bar(labels, values, ylim=(0, 1), ylabel="Accuracy", title="Few-shot classification with MNIST prototypes", legend=false)
    savefig(output_path)
end

function plot_embeddings_pca(feature_extractor, X_query, y_query, prototypes, prototype_labels; output_path="fewshot_embeddings_pca_julia.png")
    embeddings = feature_extractor(X_query)'
    proto = prototypes'

    combined = vcat(embeddings, proto)
    pca_model = fit(PCA, combined; maxoutdim=2)
    projected = transform(pca_model, combined)

    projected_query = projected[1:size(embeddings, 1), :]
    projected_proto = projected[(size(embeddings, 1)+1):end, :]

    scatter(
        projected_query[:, 1], projected_query[:, 2],
        group=y_query,
        markersize=3,
        alpha=0.6,
        xlabel="PC1",
        ylabel="PC2",
        title="PCA visualization of MNIST embeddings",
        legend=:outerright,
    )
    scatter!(projected_proto[:, 1], projected_proto[:, 2], markershape=:star5, markersize=9, label="Prototypes")

    for (i, label) in enumerate(prototype_labels)
        annotate!(projected_proto[i, 1], projected_proto[i, 2], text(string(label), 8))
    end

    savefig(output_path)
end

function main()
    X_train_known, y_train_known, X_train_seven, y_train_seven, X_test, y_test = load_mnist_world_without_sevens()

    println("Clases conocidas: ", KNOWN_CLASSES)
    println("Clase nueva: ", NOVEL_CLASS)
    println("Ejemplos de entrenamiento sin sietes: ", size(X_train_known, 4))
    println("Sietes reservados para few-shot: ", size(X_train_seven, 4))

    model = build_classifier(length(KNOWN_CLASSES))
    train_classifier!(model, X_train_known, y_train_known; epochs=3)

    known_acc = evaluate_known_classifier(model, X_test, y_test)
    println("Accuracy inicial en clases conocidas: ", round(known_acc, digits=4))

    feature_extractor = create_feature_extractor(model)

    results = Dict{String, Float64}()
    saved_for_plot = nothing

    for n_shots in [1, 5]
        acc, y_pred, y_query, X_query, prototypes, prototype_labels = run_episode(
            feature_extractor,
            X_train_known, y_train_known,
            X_train_seven, y_train_seven,
            X_test, y_test;
            n_shots=n_shots,
        )

        results["$(n_shots)-shot"] = Float64(acc)

        println("\n", repeat("=", 60))
        println("Resultados $(n_shots)-shot")
        println(repeat("=", 60))
        println("Accuracy: ", round(acc, digits=4))
        println("Matriz de confusión:")
        println(confusion_matrix(y_query, y_pred))

        if n_shots == 5
            saved_for_plot = (X_query, y_query, prototypes, prototype_labels)
        end
    end

    plot_accuracy_comparison(results)

    if saved_for_plot !== nothing
        X_query, y_query, prototypes, prototype_labels = saved_for_plot
        n = min(500, size(X_query, 4))
        sample_idx = sample(1:size(X_query, 4), n; replace=false)
        plot_embeddings_pca(
            feature_extractor,
            X_query[:, :, :, sample_idx],
            y_query[sample_idx],
            prototypes,
            prototype_labels,
        )
    end

    println("\nResumen:")
    for key in sort(collect(keys(results)))
        println("  $(key): ", round(results[key], digits=4))
    end
end

main()
