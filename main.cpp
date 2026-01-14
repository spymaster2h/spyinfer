#include <iostream>
#include <string>
#include <vector>
#include "engine/llm_engine.hpp"
#include "utils/tokenizer.hpp"



//debug
void print_tensor(const std::shared_ptr<Tensor>& tensor, int len)
{
    std::cout << "Tensor data: ";
    for (int i = 0; i < len; ++i)
    {
        if (tensor->dtype() == DataType::fp32_t)
            std::cout << tensor->data_ptr<float>()[i] << " ";
        else if (tensor->dtype() == DataType::bf16_t)
            std::cout << tensor->data_ptr<uint16_t>()[i] << " ";
        else if (tensor->dtype() == DataType::int32_t)
            std::cout << tensor->data_ptr<int>()[i] << " ";
    }
    std::cout << std::endl;
}


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];

    spyinfer::LLMEngine engine(model_path);

    std::cout << "LLM Engine initialized. Start a conversation." << std::endl;
    std::cout << "Type '/new' to start a new conversation, or '/exit' to quit." << std::endl;

    int conversation_id = -1;

    while (true)
    {
        std::cout << "\nUser: ";
        std::string user_input;
        std::getline(std::cin, user_input);

        if (user_input == "/exit")
        {
            if (conversation_id != -1) {
                engine.remove_request(conversation_id);
            }
            break;
        }

        if (user_input == "/new")
        {
            if (conversation_id != -1) {
               engine.remove_request(conversation_id);
            }
            conversation_id = -1;
            std::cout << "Starting new conversation." << std::endl;
            continue;
        }


        if (conversation_id == -1)
        {
            conversation_id = engine.add_request({{"user", user_input}});
            std::cout << "Assistant: " << std::flush;
        }
        else
        {
            // Subsequent turn in an existing conversation
            engine.append_to_request(conversation_id, {"user", user_input});
            std::cout << "Assistant: " << std::flush;
        }

        // Streaming loop
        std::string printed_output = "";
        while (!engine.is_request_finished(conversation_id))
        {
            engine.step();
            // Get only the generated part, not the full history
            std::string current_response = engine.get_output(conversation_id);
            if (current_response.length() > printed_output.length())
            {
                std::string new_text = current_response.substr(printed_output.length());
                std::cout << new_text << std::flush;
                printed_output = current_response;
            }
        }
        std::cout << std::endl;
    }

    return 0;
}