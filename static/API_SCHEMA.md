# ChatBot Agent RAG API

API сервис чат-бота поддержки с использованием RAG и LangGraph

# Base URL


| URL | Description |
|-----|-------------|


# Authentication



## Security Schemes

| Name              | Type              | Description              | Scheme              | Bearer Format             |
|-------------------|-------------------|--------------------------|---------------------|---------------------------|
| APIKeyHeader | apiKey |  |  |  |

# APIs

## POST /chat

Chat





### Request Body

[ChatRequest](#chatrequest)







### Responses

#### 200


Successful Response


[ChatResponse](#chatresponse)







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## POST /update-docs

Обновление данных в базе поиска

Эндпоинт для обновления данных в базе поиска.




### Responses

#### 200


Successful Response








## GET /health

Health Check

Проверка работоспособности сервиса.




### Responses

#### 200


Successful Response








# Components



## ChatRequest



| Field | Type | Description |
|-------|------|-------------|
| question | string | Вопрос пользователя |
| thread_id |  | Идентификатор диалога (опционально) |


## ChatResponse



| Field | Type | Description |
|-------|------|-------------|
| answer | string | Ответ чат-бота |
| thread_id |  | Идентификатор диалога |


## HTTPValidationError



| Field | Type | Description |
|-------|------|-------------|
| detail | array |  |


## ValidationError



| Field | Type | Description |
|-------|------|-------------|
| loc | array |  |
| msg | string |  |
| type | string |  |
