
🚀**Supervised Learning With Scikit-Learn - Datacamp**🚀


Repositório destinado para anotações e teste de código das aulas do curso Supervised Learning with scikit-learn do DataCamp. A maioria dos códigos são executados na própria ferramenta, mas achei válido testar algumas coisas localmente para melhor compreensão. 


📊 **O curso aborda**:
- **Criação de Modelos Preditivos**: Desenvolvimento de modelos para prever se um cliente vai deixar um serviço, identificar se um indivíduo tem diabetes e até classificar o gênero de uma música. Além da criação de Pipelines, utilizando o scikit, para Normalização dos dados e transformação com imputers.
- **Otimização e Avaliação**: no curso é trabalhado exaustivamento a utilização da validação cruzadada e é desenvolvido fine-tuning de paramâmetros dos modelos utilizando GridSearch e RandomSearch.

**Algumas discussões interessantes:**

- **Intuição do fine-tunind do parâmetro n_neighbors (número de vizinhos) do modelo k-nearest neighbors (knn):**

 O parâmetro é utilizado para definir o número de vizinhos que serão avaliados para definir a classe do novo ponto de dados, quanto maior esse número, mais o modelo se torna complexo, podendo em alguns casos melhorar a performance, mas em outros levar ao overfiting. A imagem abaixo mede a acurácia em difentes números de vizinhos para um modelo de classificação treinado e testado no dataset Iris do scikit-learn. 

(images/knn.png)

- **Utilizar o Mean Squared Error Negativo na validação cruzada:**

A validação cruzada tem como padrão escolher o maior valor na função de custo, o que é errado quando estamos usando a métrica Mean Squared Error que quanto menor é melhor, nesse caso para obtermos o melhor modelo podemos utilizar o Mean Squared Error negativo. Exemplo de aplicação: 
```python 
# Create X and y
X = music_dummies.drop('popularity',axis=1).values
y = music_dummies['popularity'].values

# Instantiate a ridge model
ridge = Ridge(alpha=0.2)

# Perform cross-validation
scores = cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_squared_error")

#Calculate RMSE
rmse = np.sqrt(-scores)
print("Average RMSE: {}".format(np.mean(rmse)))
print("Standard Deviation of the target array: {}".format(np.std(y)))

