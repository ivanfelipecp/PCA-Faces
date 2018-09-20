clear all; close all; clc; 

% ~~~~~~~~~~~~ Selecciona ranD_trainingommente que muestras agarrar ~~~~~~~~~~~~
s = 41;
samples = 10;
training_percent = 0.8;
testing_percent = 1 - training_percent;
training_samples = 1:samples;
testing_samples = [];

% Selecciona dos random samples
k = int16(samples*testing_percent);
for i=1:k
    [~, m] = size(training_samples);
    random_sample = randi([1 m],1);
    testing_samples = [testing_samples [training_samples(random_sample)]];
    training_samples(random_sample) = [];
end

% ~~~~~~~~~~~~ Leer las imágenes ~~~~~~~~~~~~
path = 'input/s';
training = [];
testing = [];

% For que itera las clases
for c=1:s
    % For que itera las muestras de training
    [~,m] = size(training_samples);
    for v=1:m
        % Obtiene el sample actual
        i = training_samples(v);
        % Se lee la image
        img = imread(strcat(path,int2str(c),'/',int2str(i),'.pgm'));
        % Se vector-columnariza
        img = img(:);
        % Se hace concatenación con training
        training = horzcat(training, img);
    end
    
    % For que itera las muestras de testing
    [~,m] = size(testing_samples);
    for v=1:m
        % Obtiene el indice del sample actual
        i = testing_samples(v);
        % Se lee la image
        img = imread(strcat(path,int2str(c),'/',int2str(i),'.pgm'));
        % Se vector-columnariza
        img = img(:);
        % Se hace concatenación con training
        testing = horzcat(testing, img);
    end
end


% ~~~~~~~~~~~~ Se calcula la matriz D para training ~~~~~~~~~~~~
[~,m] = size(training);
mu = mean(training')';
D_training = double(training) - repmat(mu, 1, m);
% ~~~~~~~~~~~~ Se calcula la matriz D para testing ~~~~~~~~~~~~
[~,m] = size(testing);
mu = mean(testing')';
D_testing = double(testing) - repmat(mu, 1, m);
% ~~~~~~~~~~~~ Se calcula la matriz covarianza ~~~~~~~~~~~~
training_sigma_v = D_training' * D_training;
% ~~~~~~~~~~~~ Se calcula auto vectores y valores de la matriz de covarianza ~~~~~~~~~~~~
[training_V, ~] = eig(training_sigma_v);
W = D_training * training_V;
% ~~~~~~~~~~~~ Se reduce W con n_prime ~~~~~~~~~~~~
% Porcentaje de muestras que se van a mantener
percent = 0.90;
[~,m] = size(W); 
n_prime = int16(m * percent);

% Suma de cada columna, para extraer los máximos
C = sum(W);
W_new = [];

for i=1:n_prime
    [~,i] = max(C);
    column_i = W(:,i);
    column_i = column_i(:);
    W_new = horzcat(W_new,column_i);
    W(:,i) = [];
    C(:,i) = [];
end

% Cambia el valor de W a la nueva que se creó
W = W_new;
% ~~~~~~~~~~~~ Matriz D_traininge caras proyectaD_trainingas ~~~~~~~~~~~~
training = W' * D_training;
testing = W' * D_testing;
% ~~~~~~~~~~~~ Se guarD_trainingan los espacios generaD_trainingos ~~~~~~~~~~~~
save(strcat('datasets_',int2str(percent*100)),'training','testing');