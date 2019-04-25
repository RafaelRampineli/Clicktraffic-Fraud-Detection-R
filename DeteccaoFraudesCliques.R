# Projeto Final 1 - Detecção de Fraudes em Cliques de Propaganda Mobile

# Legenda Variáveis Dataset

#Each row of the training data contains a click record, with the following features.

#ip: ip address of click.
#app: app id for marketing.
#device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
#os: os version id of user mobile phone
#channel: channel id of mobile ad publisher
##click_time: timestamp of click (UTC)
#attributed_time: if user download the app for after clicking an ad, this is the time of the app download
#is_attributed: the target that is to be predicted, indicating the app was downloaded

#Note that ip, app, device, os, and channel are encoded.


################ ETAPA 1: CARREGANDO O DATASET E IMPORTANDO BIBLIOTECAS NECESSÁRIAS ################

dataset <- read.csv(file = "C:/Users/rafae/OneDrive/FilestoStudy/Formacao_Cientista_Dados/BigDataRAzure/Cap20_ProjetosComFeedback/DeteccaoFraudesCliquesPropagandaMobile/dataset/train_sample.csv",
                  sep = ",")

#dataset <- read.csv(file = "C:/Users/rafael.rampineli/OneDrive/FilestoStudy/Formacao_Cientista_Dados/BigDataRAzure/Cap20_ProjetosComFeedback/DeteccaoFraudesCliquesPropagandaMobile/dataset/train_sample.csv",
#                  sep = ",")

str(dataset)

#install.packages("dplyr")
#install.packages("ggplot2")
#install.packages("DMwR")
#install.packages("randomForest")
#install.packages("caret")
#install.packages("caTools")
#install.packages("e1071")
#install.packages("ROCR")
library(dplyr)
library(ggplot2)
library(DMwR)
library(randomForest)
library(caret)
library(caTools)
library(e1071)
library(ROCR)

# Verificando se existem dados missing
any(is.na(dataset))

################ ETAPA 2: GERANDO PLOTS PARA ANÁLISE EXPLORÁTÓRIA ################

# Contagem de dados
dataset %>%
  count(is_attributed)

prop.table(table(dataset$is_attributed))

# Realizando uma análise exploratória para identificar quantidade de registros que resultaram em Download ou não.
ggplot(dataset, aes(x = dataset$is_attributed)) + 
  geom_bar(aes(fill = dataset$is_attributed)) +
  ggtitle("Clicks Downloaded / Not Downloaded") +
  xlab("Download realizado") + ylab("Quantidade")


dataset %>% filter(is_attributed == 1) %>%
ggplot(aes(x = as.Date(click_time))) + 
  geom_bar() +
  ggtitle("Quantidade de Downloads por Data")

head(dataset)

################ ETAPA 3: FEATURE ENGINEERING: TRANSFORMANDO DADOS CATEGÓRICOS E NORMALIZAÇÃO DE DADOS NUMÉRICOS  ################

# Transformando colunas categóricas
var_categoricas <- c("is_attributed") #c("ip", "app", "device", "os", "channel", "is_attributed")

func_to.factors <- function(dataset, var_categoricas){
  for (n in var_categoricas){
    dataset[[n]] <- as.factor(dataset[[n]])
  }
  return(dataset)
}

dataset_formated <- func_to.factors(dataset, var_categoricas)

# Normalizando os dados numéricos
vars_to_scale <- c("ip", "app", "device", "os", "channel")

func_scale.features <- function(dataset, vars_to_scale){
  for (n in vars_to_scale){
    dataset[[n]] <- scale(dataset[[n]], center=T, scale=T) # Função scale está sendo utilizada para realizar a normalização dos dados
  }
  return(dataset)
}

dataset_formated <- func_scale.features(dataset_formated, vars_to_scale)

str(dataset_formated)

# Formatando as colunas de data
dataset_formated$click_time <- as.factor(as.Date(dataset_formated$click_time))
dataset_formated$attributed_time <- NULL

final_dataset <- dataset_formated

################ ETAPA 4: ANALISANDO E APLICANDO FEATURE SELECTION ################ 

# Utilizando o algoritmo randomForest para Obter as variaveis mais relevantes do dataset
importance_vars <- randomForest(is_attributed ~ ., 
                                data = final_dataset, 
                                ntree = 100, 
                                nodesize = 10, 
                                importance = TRUE)

varImp(importance_vars)
importance_vars

################ ETAPA 5: SPLIT DOS DADOS EM TREINO E TESTE E CRIAÇÃO DO MODELO UTILIZANDO O ALGORITMO LINEAR LOGISTIC ################ 

# Split dados em Treino e Teste
trainindex <- sample.split(final_dataset$is_attributed, SplitRatio = 0.65)
dados_treino <- subset(final_dataset, trainindex == TRUE)
dados_teste <- subset(final_dataset, trainindex == FALSE)

################ ETAPA 6: CRIANDO MODELOS ################ 

# Modelo 1: Criando um modelo utilizando regressão Logistica

# O resultado do modelo de Regressão Logística consiste em "probabilidade" de um evento acontecer devido ao "response".
# O resultado varia entre 0-1 e iremos realizar um arredondamento nos valores para os valores serem 0 ou 1,
# para posteriormente ser possível realizar a comparação do resultado através do método confusionMatrix.
linear_logistic_model <- glm(is_attributed ~ ., 
                             data = dados_treino, 
                             family = binomial(link = "logit"))

linear_logistic_predict <- predict(linear_logistic_model, 
                                   newdata = dados_teste, 
                                   type = "response")

linear_logistic_predict <- round(linear_logistic_predict)

confusionMatrix(table(data = linear_logistic_predict, 
                      reference = dados_teste[,7]), 
                positive = '1')


# Resultado accuracy Modelo 1: 0,9977 Modelo acertou quase todas as classificações resultantes em 0 porém errou todas com 1.

# Modelo 2: Criando um modelo utilizando Random Forest

random_forest_model <- randomForest(is_attributed ~ ., 
                                    data = dados_treino, 
                                    method = "class")

radomForest_predict <- predict(random_forest_model, 
                               newdata = dados_teste, 
                               type = "class")

confusionMatrix(table(data = radomForest_predict, 
                      reference = dados_teste[,7]), 
                positive = '1')

# Resultado accuracy Modelo 2: 0,998 Modelo melhorou a precisão para resultados 1.

# É possível analisar através da confusionMatrix, que os algoritmos conseguiram resolver muito bem os registros que onde a classificação final é igual 0.
# Essa informação pode ser visto em "Sensitivity" e "Specificity".
# Isso ocorreu por causa do volume de dados no dataset ser tendêncioso em relação a classificação 0.

# Para tentar solucionar esse problema, irei aplicar a ingestão de dados utizando metodos SMOTE e ROSE.

################ ETAPA 4: REALIZANDO O BALANCEAMENTO UTILIZANDO SMOTE ################ 

# Balanceando os dados 
final_dataset_Smotted <- SMOTE(is_attributed ~ . , final_dataset, perc.over = 20000,perc.under=200)

prop.table(table(final_dataset_Smotted$is_attributed))
table(final_dataset_Smotted$is_attributed)

################ ETAPA 5: APLICAR O SPLIT DOS DADOS EM TREINO E TESTE NOVAMENTE ################ 

# Split dados em Treino e Teste
trainindex_Smotted <- sample.split(final_dataset_Smotted$is_attributed, SplitRatio = 0.65)
dados_treino_Smotted <- subset(final_dataset_Smotted, trainindex == TRUE)
dados_teste_Smotted <- subset(final_dataset_Smotted, trainindex == FALSE)


linear_logistic_model_smotted <- glm(is_attributed ~ ., 
                             data = dados_treino_Smotted, 
                             family = binomial(link = "logit"))

linear_logistic_predict_smotted <- predict(linear_logistic_model_smotted, 
                                   newdata = dados_teste_Smotted, 
                                   type = "response")

linear_logistic_predict_smotted <- round(linear_logistic_predict_smotted)

confusionMatrix(table(data = linear_logistic_predict_smotted, 
                      reference = dados_teste_Smotted[,7]), 
                positive = '1')

# Resultado accuracy Modelo 1: 0,8169 A accuracy abaixou, porém melhorou a taxa de acerto resultantes em 0 e 1.

# Modelo 2: Criando um modelo utilizando Random Forest

random_forest_model_smotted <- randomForest(is_attributed ~ ., 
                                    data = dados_treino_Smotted, 
                                    method = "class")

radomForest_predict_smotted <- predict(random_forest_model_smotted, 
                               newdata = dados_teste_Smotted, 
                               type = "class")

confusionMatrix(table(data = radomForest_predict_smotted, 
                      reference = dados_teste_Smotted[,7]), 
                positive = '1')

# Resultado accuracy Modelo 2: 0,9965 com uma excelente acertividade para ambas classificações. 


################ ETAPA 6: GERANDO UM GRÁFICO DE CURVA ROC ################ 

# Gerando previsões nos dados de teste
# Criando um dataframe com os dados ORIGINAIS do TESTE junto com a previsão do modelo utilizando os dados de teste.
# com esse dataset poderemos criar uma confusion matrix para analisar a taxa de acerto do modelo de classificação randomForest.
df_previsoes_smotted <- data.frame(observado = dados_teste_Smotted$is_attributed,
                           previsto = radomForest_predict_smotted)


pred <- prediction(as.numeric(df_previsoes_smotted$previsto), as.numeric(df_previsoes_smotted$observado))
perf <- performance(pred, "tpr","fpr") 
plot(perf, col = rainbow(10))

