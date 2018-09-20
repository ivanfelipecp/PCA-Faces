clear all; close all; clc; 
% ~~~~~~~~~~~~ Carga de datos ~~~~~~~~~~~~
training_file = "datasets_90";
load(training_file,'training','testing');
% ~~~~~~~~~~~~ Crea el vector de datos ~~~~~~~~~~~~
training_data = training;
testing_data = testing;
% ~~~~~~~~~~~~ Crea el vector de etiquetas para training~~~~~~~~~~~~
s = 41;
training_labels = 1:s;
% ~~~~~~~~~~~~ Crea el vector de etiquetas para testing~~~~~~~~~~~~
samples = 10;
percent_training = 0.2;
testing_labels = [];
for i=1:s
    temp = [i];
    testing_labels = horzcat(testing_labels,repelem(temp,samples*percent_training));
end

% ~~~~~~~~~~~~ Crea el vector de centroides ~~~~~~~~~~~~
U  = [];
percent_training = 0.8;
n_samples = samples*percent_training - 1;
for i=1:s
    index = int16(samples*percent_training) * i;
    actual_class = training_data(:,index - n_samples:index);
    u = mean(actual_class')';
    U = horzcat(U,u);
end

training_data = U;

% ~~~~~~~~~~~~ Testing ~~~~~~~~~~~~
distance = 'euclidean';
euc_acc = do_centroid(training_data, training_labels, testing_data, testing_labels, distance);

distance = 'manhattan';
man_acc = do_centroid(training_data, training_labels, testing_data, testing_labels, distance);

function y = do_centroid(training_data, training_labels, testing_data, testing_labels, distance)
    [~,m] = size(testing_data);
    predictions = zeros(1,m);
    % Recorre todos los test_items en test_data
    for i=1:m
        testing_item = testing_data(:,i);
        predict = get_closest_centroid(training_data, training_labels, testing_item, distance);
        predictions(1,i) = predict;
    end
    
    y = sum(predictions==testing_labels)/m;
end


function p = get_closest_centroid(training_data, training_labels, testing_item, distance)
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
    
    [~,index] = min(distances);
    p = training_labels(index);
end

function y = euclidean_distance(A,B)
    y = norm(A - B);
end

function y = manhattan_distance(A,B)
    y = sum(abs(A - B));
end
