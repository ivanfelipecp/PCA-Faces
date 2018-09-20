clear all; close all; clc; 
% ~~~~~~~~~~~~ Carga de datos ~~~~~~~~~~~~
training_file = "datasets_90";
load(training_file,'training','testing');
% ~~~~~~~~~~~~ Crea el vector de datos ~~~~~~~~~~~~
training_data = training;
testing_data = testing;
% ~~~~~~~~~~~~ Crea el vector de etiquetas ~~~~~~~~~~~~
training_labels = [];
testing_labels = [];
samples = 10;
s = 41;
percent_training = 0.8;
percent_testing = 0.2;

for i=1:s
    temp = [i];
    training_labels = horzcat(training_labels,repelem(temp,samples*percent_training));
    testing_labels = horzcat(testing_labels,repelem(temp,samples*percent_testing));
end

distance = 'euclidean';
K = 3;
eu_acc3 = do_KNN(training_data, training_labels, testing_data, testing_labels, K, distance);

K = 5;
eu_acc5 = do_KNN(training_data, training_labels, testing_data, testing_labels, K, distance);

K = 7;
eu_acc7 = do_KNN(training_data, training_labels, testing_data, testing_labels, K, distance);

K = 10;
eu_acc10 = do_KNN(training_data, training_labels, testing_data, testing_labels, K, distance);

distance = 'manhattan';
K = 3;
mann_acc3 = do_KNN(training_data, training_labels, testing_data, testing_labels, K, distance);

K = 5;
mann_acc5 = do_KNN(training_data, training_labels, testing_data, testing_labels, K, distance);

K = 7;
mann_acc7 = do_KNN(training_data, training_labels, testing_data, testing_labels, K, distance);

K = 10;
man_acc10 = do_KNN(training_data, training_labels, testing_data, testing_labels, K, distance);

function y = do_KNN(training_data, training_labels, testing_data, testing_labels, K, distance)
    [~,m] = size(testing_data);
    predictions = zeros(1,m);
    % Recorre todos los test_items en test_data
    for i=1:m
        testing_item = testing_data(:,i);
        predict = KNN(training_data, training_labels, testing_item, K, distance);
        predictions(1,i) = predict;
    end
    
    y = sum(predictions==testing_labels)/m;
end

function p = KNN(training_data, training_labels, testing_item, K, distance)
    [~,m] = size(training_data);
    distances = zeros(1,m);
    
    % Calcula las distancias entre todos los puntos del espacio y el item
    % actual
    for i=1:m
        switch distance
            case 'euclidean'
                distances(1,i) = euclidean_distance(training_data(:,i), testing_item);
            case 'manhattan'
                distances(1,i) = manhattan_distance(training_data(:,i), testing_item);
        end
    end
    
    % Extrae los k vecinos más cercanos
    neighbors = zeros(1,K);
    for i=1:K
        [~,index] = min(distances);
        neighbors(1,i) = training_labels(index);
        distances(index) = [];
        training_labels(index) = [];
    end
    
    p = mode(neighbors);
end

function y = euclidean_distance(A,B)
    y = norm(A - B);
end

function y = manhattan_distance(A,B)
    y = sum(abs(A - B));
end