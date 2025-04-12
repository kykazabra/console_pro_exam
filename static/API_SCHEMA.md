# ChatBot Agent RAG API

API ������ ���-���� ��������� � �������������� RAG � LangGraph

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

���������� ������ � ���� ������

�������� ��� ���������� ������ � ���� ������.




### Responses

#### 200


Successful Response








## GET /health

Health Check

�������� ����������������� �������.




### Responses

#### 200


Successful Response








# Components



## ChatRequest



| Field | Type | Description |
|-------|------|-------------|
| question | string | ������ ������������ |
| thread_id |  | ������������� ������� (�����������) |


## ChatResponse



| Field | Type | Description |
|-------|------|-------------|
| answer | string | ����� ���-���� |
| thread_id |  | ������������� ������� |


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
