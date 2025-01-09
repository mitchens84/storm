**User:**

Task: "where can i customise the LLMs i want to use?"

---

**Assistant:**

To answer this question effectively, I'll need to examine the project structure and look for relevant configuration files or code that might be related to LLM customization. Let's start by getting an overview of the project files.
[Tool Use: list_files_recursive]
Path: .

---

**User:**

[Tool Result]
assets/
eval/
examples/
examples/helper/
FreshWiki/
FreshWiki/json/
FreshWiki/txt/
frontend/
frontend/demo_light/
frontend/demo_light/assets/
frontend/demo_light/pages_util/
knowledge_storm/
knowledge_storm/storm_wiki/
knowledge_storm/storm_wiki/modules/
.gitignore
assets/logo.svg
assets/overview.svg
assets/storm_naacl2024_slides.pdf
assets/two_stages.jpg
eval/citation_quality.py
eval/eval_article_quality.py
eval/eval_outline_quality.py
eval/eval_prometheus_no_ref.prompt
eval/eval_rubric_5.json
eval/evaluation_prometheus.py
eval/evaluation_trim_length.py
eval/metrics.py
examples/helper/process_kaggle_arxiv_abstract_dataset.py
examples/README.md
examples/run_storm_wiki_claude.py
examples/run_storm_wiki_gpt_with_VectorRM.py
examples/run_storm_wiki_gpt.py
examples/run_storm_wiki_mistral.py
examples/run_storm_wiki_ollama.py
FreshWiki/get_fresh_wiki_page.py
FreshWiki/json/65th_Annual_Grammy_Awards.json
FreshWiki/json/2022_AFL_Grand_Final.json
FreshWiki/json/2022_Crimean_Bridge_explosion.json
FreshWiki/json/2022_Hungarian_parliamentary_election.json
FreshWiki/json/2022_Istanbul_bombing.json
FreshWiki/json/2022_Luzon_earthquake.json
FreshWiki/json/2022_New_York_City_Subway_attack.json
FreshWiki/json/2022_Pakistan_floods.json
FreshWiki/json/2022_UEFA_Champions_League_final.json
FreshWiki/json/2022_United_States_Senate_election_in_Pennsylvania.json
FreshWiki/json/2022_Welsh_Open_(snooker).json
FreshWiki/json/2022_West_Java_earthquake.json
FreshWiki/json/2022_Winter_Olympics_opening_ceremony.json
FreshWiki/json/2022_World_Figure_Skating_Championships.json
FreshWiki/json/2023_Emilia-Romagna_floods.json
FreshWiki/json/2023_Monterey_Park_shooting.json
FreshWiki/json/2023_Odisha_train_collision.json
FreshWiki/json/2023_SEA_Games.json
FreshWiki/json/2023_Tour_de_France.json
FreshWiki/json/2023_Wagner_Group_plane_crash.json
FreshWiki/json/African_Greeks.json
FreshWiki/json/Archie_Battersbee_case.json
FreshWiki/json/Argentina_national_football_team.json
FreshWiki/json/Armorial_of_Albania.json
FreshWiki/json/Banjska_attack.json
FreshWiki/json/Barbie__The_Album.json
FreshWiki/json/Candidates_Tournament_2022.json
FreshWiki/json/China_Eastern_Airlines_Flight_5735.json
FreshWiki/json/Chris_Sharma.json
FreshWiki/json/Christian_Atsu.json
FreshWiki/json/Cyclone_Batsirai.json
FreshWiki/json/Cyclone_Mocha.json
FreshWiki/json/Dancing_with_the_Stars_(American_season_31).json
FreshWiki/json/Darya_Dugina.json
FreshWiki/json/Death_and_funeral_of_Pope_Benedict_XVI.json
FreshWiki/json/Dumas_Malone.json
FreshWiki/json/E.A.T._(TV_program).json
FreshWiki/json/Edwards_Vacuum.json
FreshWiki/json/Eukaryote.json
FreshWiki/json/Eurovision_Song_Contest_2024.json
FreshWiki/json/Fantastic_Beasts__The_Secrets_of_Dumbledore.json
FreshWiki/json/Febrile_seizure.json
FreshWiki/json/Funeral_for_Yesterday.json
FreshWiki/json/Gehraiyaan.json
FreshWiki/json/George_Young_(swimmer).json
FreshWiki/json/Georgia_Guidestones.json
FreshWiki/json/Good_News_International_Ministries.json
FreshWiki/json/Green_Party_of_California.json
FreshWiki/json/Herder_Memorial_Trophy.json
FreshWiki/json/Hessisches_Landesmuseum_Darmstadt.json
FreshWiki/json/Hurricane_Hilary.json
FreshWiki/json/Hurricane_Idalia.json
FreshWiki/json/Illegal_operation_(euphemism).json
FreshWiki/json/Irene_Mawer.json
FreshWiki/json/Jackass_Forever.json
FreshWiki/json/James_Davis_(printer).json
FreshWiki/json/Joe_Biden_classified_documents_incident.json
FreshWiki/json/John_Harrison_Stonehouse.json
FreshWiki/json/July_2022_United_Kingdom_government_crisis.json
FreshWiki/json/Kenkoku_University.json
FreshWiki/json/Killing_of_Amir_Locke.json
FreshWiki/json/Lahaina,_Hawaii.json
FreshWiki/json/Liga_1_(Indonesia).json
FreshWiki/json/LK-99.json
FreshWiki/json/Luttra_Woman.json
FreshWiki/json/Mass_and_Cass.json
FreshWiki/json/Michelle_Yeoh.json
FreshWiki/json/Minions__The_Rise_of_Gru.json
FreshWiki/json/Miss_Grand_International_2022.json
FreshWiki/json/Miss_Universe_2022.json
FreshWiki/json/Mount_Greylock.json
FreshWiki/json/Music_written_in_all_major_and_or_minor_keys.json
FreshWiki/json/Nagpuria_people.json
FreshWiki/json/O'Sulloc_Tea.json
FreshWiki/json/OceanGate.json
FreshWiki/json/October_2022_United_Kingdom_government_crisis.json
FreshWiki/json/Pinoy_Big_Brother__Kumunity_Season_10.json
FreshWiki/json/Pishdadian_dynasty.json
FreshWiki/json/Queue_for_the_lying-in-state_of_Elizabeth_II.json
FreshWiki/json/Residential_Palace_Darmstadt.json
FreshWiki/json/Rock-climbing_equipment.json
FreshWiki/json/Sardar_(2022_film).json
FreshWiki/json/Shepherd_Building_Group.json
FreshWiki/json/Silicon_Valley_Bank.json
FreshWiki/json/SpaceX_Starship_Integrated_Flight_Test.json
FreshWiki/json/Speak_Now_(Taylor's_Version).json
FreshWiki/json/Storm_Eunice.json
FreshWiki/json/Takeoff_(rapper).json
FreshWiki/json/Tartus.json
FreshWiki/json/Taylor_Hawkins.json
FreshWiki/json/The_Book_of_Boba_Fett.json
FreshWiki/json/The_Northman.json
FreshWiki/json/The_Papers_of_Benjamin_Franklin.json
FreshWiki/json/Those_Once_Loyal.json
FreshWiki/json/Threads_(social_network).json
FreshWiki/json/Top-four_primary.json
FreshWiki/json/Treat_Williams.json
FreshWiki/json/Typhoon_Hinnamnor.json
FreshWiki/json/Vande_Bharat_Express.json
FreshWiki/json/Zillennials.json
FreshWiki/LICENSE
FreshWiki/README.md
FreshWiki/topic_list.csv
FreshWiki/txt/65th_Annual_Grammy_Awards.txt
FreshWiki/txt/2022_AFL_Grand_Final.txt
FreshWiki/txt/2022_Crimean_Bridge_explosion.txt
FreshWiki/txt/2022_Hungarian_parliamentary_election.txt
FreshWiki/txt/2022_Istanbul_bombing.txt
FreshWiki/txt/2022_Luzon_earthquake.txt
FreshWiki/txt/2022_New_York_City_Subway_attack.txt
FreshWiki/txt/2022_Pakistan_floods.txt
FreshWiki/txt/2022_UEFA_Champions_League_final.txt
FreshWiki/txt/2022_United_States_Senate_election_in_Pennsylvania.txt
FreshWiki/txt/2022_Welsh_Open_(snooker).txt
FreshWiki/txt/2022_West_Java_earthquake.txt
FreshWiki/txt/2022_Winter_Olympics_opening_ceremony.txt
FreshWiki/txt/2022_World_Figure_Skating_Championships.txt
FreshWiki/txt/2023_Emilia-Romagna_floods.txt
FreshWiki/txt/2023_Monterey_Park_shooting.txt
FreshWiki/txt/2023_Odisha_train_collision.txt
FreshWiki/txt/2023_SEA_Games.txt
FreshWiki/txt/2023_Tour_de_France.txt
FreshWiki/txt/2023_Wagner_Group_plane_crash.txt
FreshWiki/txt/African_Greeks.txt
FreshWiki/txt/Archie_Battersbee_case.txt
FreshWiki/txt/Argentina_national_football_team.txt
FreshWiki/txt/Armorial_of_Albania.txt
FreshWiki/txt/Banjska_attack.txt
FreshWiki/txt/Barbie__The_Album.txt
FreshWiki/txt/Candidates_Tournament_2022.txt
FreshWiki/txt/China_Eastern_Airlines_Flight_5735.txt
FreshWiki/txt/Chris_Sharma.txt
FreshWiki/txt/Christian_Atsu.txt
FreshWiki/txt/Cyclone_Batsirai.txt
FreshWiki/txt/Cyclone_Mocha.txt
FreshWiki/txt/Dancing_with_the_Stars_(American_season_31).txt
FreshWiki/txt/Darya_Dugina.txt
FreshWiki/txt/Death_and_funeral_of_Pope_Benedict_XVI.txt
FreshWiki/txt/Dumas_Malone.txt
FreshWiki/txt/E.A.T._(TV_program).txt
FreshWiki/txt/Edwards_Vacuum.txt
FreshWiki/txt/Eukaryote.txt
FreshWiki/txt/Eurovision_Song_Contest_2024.txt
FreshWiki/txt/Fantastic_Beasts__The_Secrets_of_Dumbledore.txt
FreshWiki/txt/Febrile_seizure.txt
FreshWiki/txt/Funeral_for_Yesterday.txt
FreshWiki/txt/Gehraiyaan.txt
FreshWiki/txt/George_Young_(swimmer).txt
FreshWiki/txt/Georgia_Guidestones.txt
FreshWiki/txt/Good_News_International_Ministries.txt
FreshWiki/txt/Green_Party_of_California.txt
FreshWiki/txt/Herder_Memorial_Trophy.txt
FreshWiki/txt/Hessisches_Landesmuseum_Darmstadt.txt
FreshWiki/txt/Hurricane_Hilary.txt
FreshWiki/txt/Hurricane_Idalia.txt
FreshWiki/txt/Illegal_operation_(euphemism).txt
FreshWiki/txt/Irene_Mawer.txt
FreshWiki/txt/Jackass_Forever.txt
FreshWiki/txt/James_Davis_(printer).txt
FreshWiki/txt/Joe_Biden_classified_documents_incident.txt
FreshWiki/txt/John_Harrison_Stonehouse.txt
FreshWiki/txt/July_2022_United_Kingdom_government_crisis.txt
FreshWiki/txt/Kenkoku_University.txt
FreshWiki/txt/Killing_of_Amir_Locke.txt
FreshWiki/txt/Lahaina,_Hawaii.txt
FreshWiki/txt/Liga_1_(Indonesia).txt
FreshWiki/txt/LK-99.txt
FreshWiki/txt/Luttra_Woman.txt
FreshWiki/txt/Mass_and_Cass.txt
FreshWiki/txt/Michelle_Yeoh.txt
FreshWiki/txt/Minions__The_Rise_of_Gru.txt
FreshWiki/txt/Miss_Grand_International_2022.txt
FreshWiki/txt/Miss_Universe_2022.txt
FreshWiki/txt/Mount_Greylock.txt
FreshWiki/txt/Music_written_in_all_major_and_or_minor_keys.txt
FreshWiki/txt/Nagpuria_people.txt
FreshWiki/txt/O'Sulloc_Tea.txt
FreshWiki/txt/OceanGate.txt
FreshWiki/txt/October_2022_United_Kingdom_government_crisis.txt
FreshWiki/txt/Pinoy_Big_Brother__Kumunity_Season_10.txt
FreshWiki/txt/Pishdadian_dynasty.txt
FreshWiki/txt/Queue_for_the_lying-in-state_of_Elizabeth_II.txt
FreshWiki/txt/Residential_Palace_Darmstadt.txt
FreshWiki/txt/Rock-climbing_equipment.txt
FreshWiki/txt/Sardar_(2022_film).txt
FreshWiki/txt/Shepherd_Building_Group.txt
FreshWiki/txt/Silicon_Valley_Bank.txt
FreshWiki/txt/SpaceX_Starship_Integrated_Flight_Test.txt
FreshWiki/txt/Speak_Now_(Taylor's_Version).txt
FreshWiki/txt/Storm_Eunice.txt
FreshWiki/txt/Takeoff_(rapper).txt
FreshWiki/txt/Tartus.txt
FreshWiki/txt/Taylor_Hawkins.txt
FreshWiki/txt/The_Book_of_Boba_Fett.txt
FreshWiki/txt/The_Northman.txt
FreshWiki/txt/The_Papers_of_Benjamin_Franklin.txt
FreshWiki/txt/Those_Once_Loyal.txt
FreshWiki/txt/Threads_(social_network).txt
FreshWiki/txt/Top-four_primary.txt
FreshWiki/txt/Treat_Williams.txt
FreshWiki/txt/Typhoon_Hinnamnor.txt
FreshWiki/txt/Vande_Bharat_Express.txt
FreshWiki/txt/Zillennials.txt
FreshWiki/wikipage_extractor.py
frontend/demo_light/assets/article_display.jpg
frontend/demo_light/assets/create_article.jpg
frontend/demo_light/assets/void.jpg
frontend/demo_light/demo_util.py
frontend/demo_light/pages_util/CreateNewArticle.py
frontend/demo_light/pages_util/MyArticles.py
frontend/demo_light/README.md
frontend/demo_light/requirements.txt
frontend/demo_light/stoc.py
frontend/demo_light/storm.py
knowledge_storm/__init__.py
knowledge_storm/interface.py
knowledge_storm/lm.py
knowledge_storm/rm.py
knowledge_storm/storm_wiki/__init__.py
knowledge_storm/storm_wiki/engine.py
knowledge_storm/storm_wiki/modules/__init__.py
knowledge_storm/storm_wiki/modules/article_generation.py
knowledge_storm/storm_wiki/modules/article_polish.py
knowledge_storm/storm_wiki/modules/callback.py
knowledge_storm/storm_wiki/modules/knowledge_curation.py
knowledge_storm/storm_wiki/modules/outline_generation.py
knowledge_storm/storm_wiki/modules/persona_generator.py
knowledge_storm/storm_wiki/modules/retriever.py
knowledge_storm/storm_wiki/modules/storm_dataclass.py
knowledge_storm/utils.py
LICENSE
README.md
requirements.txt
setup.py

---

**Assistant:**

Based on the project structure, it appears that the LLM customization might be handled in the `knowledge_storm` directory, specifically in the `lm.py` file. Let's examine the contents of this file to see if it contains information about customizing LLMs.
[Tool Use: read_file]
Path: knowledge_storm/lm.py

---

**User:**

[Tool Result]
import logging
import os
import random
import threading
from typing import Optional, Literal, Any

import backoff
import dspy
import requests
from dsp import ERRORS, backoff_hdlr, giveup_hdlr
from dsp.modules.hf import openai_to_hf
from dsp.modules.hf_client import send_hfvllm_request_v00, send_hftgi_request_v01_wrapped
from transformers import AutoTokenizer

try:
    from anthropic import RateLimitError
except ImportError:
    RateLimitError = None


class OpenAIModel(dspy.OpenAI):
    """A wrapper class for dspy.OpenAI."""

    def __init__(
            self,
            model: str = "gpt-3.5-turbo-instruct",
            api_key: Optional[str] = None,
            model_type: Literal["chat", "text"] = None,
            **kwargs
    ):
        super().__init__(model=model, api_key=api_key, model_type=model_type, **kwargs)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get('usage')
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('prompt_tokens', 0)
                self.completion_tokens += usage_data.get('completion_tokens', 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get('model') or self.kwargs.get('engine'):
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        # if kwargs.get("n", 1) > 1:
        #     if self.model_type == "chat":
        #         kwargs = {**kwargs}
        #     else:
        #         kwargs = {**kwargs, "logprobs": 5}

        response = self.request(prompt, **kwargs)

        # Log the token usage from the OpenAI API response.
        self.log_usage(response)

        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c["logprobs"]["tokens"],
                    c["logprobs"]["token_logprobs"],
                )

                if "<|endoftext|>" in tokens:
                    index = tokens.index("<|endoftext|>") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c)))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions


class AzureOpenAIModel(dspy.AzureOpenAI):
    """A wrapper class for dspy.AzureOpenAI."""
    def __init__(
            self,
            api_base: Optional[str] = None,
            api_version: Optional[str] = None,
            model: str = "gpt-3.5-turbo-instruct",
            api_key: Optional[str] = None,
            model_type: Literal["chat", "text"] = "chat",
            **kwargs,
    ):
        super().__init__(
            api_base=api_base, api_version=api_version, model=model, api_key=api_key, model_type=model_type, **kwargs)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response.
        Override log_usage() in dspy.AzureOpenAI for tracking accumulated token usage."""
        usage_data = response.get('usage')
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('prompt_tokens', 0)
                self.completion_tokens += usage_data.get('completion_tokens', 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get('model') or self.kwargs.get('engine'):
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage


class ClaudeModel(dspy.dsp.modules.lm.LM):
    """Copied from dspy/dsp/modules/anthropic.py with the addition of tracking token usage."""

    def __init__(
            self,
            model: str,
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(model)
        try:
            from anthropic import Anthropic
        except ImportError as err:
            raise ImportError("Claude requires `pip install anthropic`.") from err

        self.provider = "anthropic"
        self.api_key = api_key = os.environ.get("ANTHROPIC_API_KEY") if api_key is None else api_key
        self.api_base = "https://api.anthropic.com/v1/messages" if api_base is None else api_base
        self.kwargs = {"temperature": kwargs.get("temperature", 0.0),
                       "max_tokens": min(kwargs.get("max_tokens", 4096), 4096), "top_p": kwargs.get("top_p", 1.0),
                       "top_k": kwargs.get("top_k", 1), "n": kwargs.pop("n", kwargs.pop("num_generations", 1)),
                       **kwargs, "model": model}
        self.history: list[dict[str, Any]] = []
        self.client = Anthropic(api_key=api_key)
        self.model = model

        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the Anthropic API response."""
        usage_data = response.usage
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.input_tokens
                self.completion_tokens += usage_data.output_tokens

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.model:
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        # caching mechanism requires hashable kwargs
        kwargs["messages"] = [{"role": "user", "content": prompt}]
        kwargs.pop("n")
        response = self.client.messages.create(**kwargs)
        # history = {
        #     "prompt": prompt,
        #     "response": response,
        #     "kwargs": kwargs,
        #     "raw_kwargs": raw_kwargs,
        # }
        json_serializable_history = {
            "prompt": prompt,
            "response": {
                "content": response.content[0].text,
                "model": response.model,
                "role": response.role,
                "stop_reason": response.stop_reason,
                "stop_sequence": response.stop_sequence,
                "type": response.type,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            },
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(json_serializable_history)
        return response

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError,),
        max_time=1000,
        max_tries=8,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Anthropic whilst handling API errors."""
        return self.basic_request(prompt, **kwargs)

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        """Retrieves completions from Anthropic.

        Args:
            prompt (str): prompt to send to Anthropic
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[str]: list of completion choices
        """
        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        # per eg here: https://docs.anthropic.com/claude/reference/messages-examples
        # max tokens can be used as a proxy to return smaller responses
        # so this cannot be a proper indicator for incomplete response unless it isnt the user-intent.
        n = kwargs.pop("n", 1)
        completions = []
        for _ in range(n):
            response = self.request(prompt, **kwargs)
            self.log_usage(response)
            # This is the original behavior in dspy/dsp/modules/anthropic.py.
            # Comment it out because it can cause "IndexError: list index out of range" silently
            # which is not transparent to developers.
            # if only_completed and response.stop_reason == "max_tokens":
            #     continue
            completions = [c.text for c in response.content]
        return completions


class VLLMClient(dspy.HFClientVLLM):
    """A wrapper class for dspy.HFClientVLLM."""

    def __init__(self, model, port, url="http://localhost", **kwargs):
        """Copied from dspy/dsp/modules/hf_client.py with the addition of storing additional kwargs."""

        super().__init__(model=model, port=port, url=url, **kwargs)
        # Store additional kwargs for the generate method.
        self.kwargs = {**self.kwargs, **kwargs}

    def _generate(self, prompt, **kwargs):
        """Copied from dspy/dsp/modules/hf_client.py with the addition of passing kwargs to VLLM server."""
        kwargs = {**self.kwargs, **kwargs}

        # payload = {
        #     "model": kwargs["model"],
        #     "prompt": prompt,
        #     "max_tokens": kwargs["max_tokens"],
        #     "temperature": kwargs["temperature"],
        # }
        payload = {
            "prompt": prompt,
            **kwargs
        }

        response = send_hfvllm_request_v00(
            f"{self.url}/v1/completions",
            json=payload,
            headers=self.headers,
        )

        try:
            json_response = response.json()
            completions = json_response["choices"]
            response = {
                "prompt": prompt,
                "choices": [{"text": c["text"]} for c in completions],
            }
            return response

        except Exception as e:
            print("Failed to parse JSON response:", response.text)
            raise Exception("Received invalid JSON response from server")


class OllamaClient(dspy.OllamaLocal):
    """A wrapper class for dspy.OllamaClient."""

    def __init__(self, model, port, url="http://localhost", **kwargs):
        """Copied from dspy/dsp/modules/hf_client.py with the addition of storing additional kwargs."""
        # Check if the URL has 'http://' or 'https://'
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url
        super().__init__(model=model, base_url=f"{url}:{port}", **kwargs)
        # Store additional kwargs for the generate method.
        self.kwargs = {**self.kwargs, **kwargs}
    

class TGIClient(dspy.HFClientTGI):
    def __init__(self, model, port, url, http_request_kwargs=None, **kwargs):
        super().__init__(model=model, port=port, url=url, http_request_kwargs=http_request_kwargs, **kwargs)

    def _generate(self, prompt, **kwargs):
        """Copied from dspy/dsp/modules/hf_client.py with the addition of removing hard-coded parameters."""
        kwargs = {**self.kwargs, **kwargs}

        payload = {
            "inputs": prompt,
            "parameters": {
                "do_sample": kwargs["n"] > 1,
                "best_of": kwargs["n"],
                "details": kwargs["n"] > 1,
                **kwargs,
            },
        }

        payload["parameters"] = openai_to_hf(**payload["parameters"])

        # Comment out the following lines to remove the hard-coded parameters.
        # payload["parameters"]["temperature"] = max(
        #     0.1, payload["parameters"]["temperature"],
        # )

        response = send_hftgi_request_v01_wrapped(
            f"{self.url}:{random.Random().choice(self.ports)}" + "/generate",
            url=self.url,
            ports=tuple(self.ports),
            json=payload,
            headers=self.headers,
            **self.http_request_kwargs,
        )

        try:
            json_response = response.json()
            # completions = json_response["generated_text"]

            completions = [json_response["generated_text"]]

            if (
                    "details" in json_response
                    and "best_of_sequences" in json_response["details"]
            ):
                completions += [
                    x["generated_text"]
                    for x in json_response["details"]["best_of_sequences"]
                ]

            response = {"prompt": prompt, "choices": [{"text": c} for c in completions]}
            return response
        except Exception:
            print("Failed to parse JSON response:", response.text)
            raise Exception("Received invalid JSON response from server")


class TogetherClient(dspy.HFModel):
    """A wrapper class for dspy.Together."""

    def __init__(self, model, apply_tokenizer_chat_template=False, hf_tokenizer_name=None, **kwargs):
        """Copied from dspy/dsp/modules/hf_client.py with the support of applying tokenizer chat template."""

        super().__init__(model=model, is_client=True)
        self.session = requests.Session()
        self.api_base = "https://api.together.xyz/v1/completions" if os.getenv(
            "TOGETHER_API_BASE") is None else os.getenv("TOGETHER_API_BASE")
        self.token = os.getenv("TOGETHER_API_KEY")
        self.model = model

        # self.use_inst_template = False
        # if any(keyword in self.model.lower() for keyword in ["inst", "instruct"]):
        #     self.use_inst_template = True
        self.apply_tokenizer_chat_template = apply_tokenizer_chat_template
        if self.apply_tokenizer_chat_template:
            logging.info("Loading huggingface tokenizer.")
            if hf_tokenizer_name is None:
                hf_tokenizer_name = self.model
            self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name, cache_dir=kwargs.get("cache_dir", None))

        stop_default = "\n\n---"

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 512,
            "top_p": 1,
            "top_k": 20,
            "repetition_penalty": 1,
            "n": 1,
            "stop": stop_default if "stop" not in kwargs else kwargs["stop"],
            **kwargs,
        }
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get('usage')
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('prompt_tokens', 0)
                self.completion_tokens += usage_data.get('completion_tokens', 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.model:
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    def _generate(self, prompt, use_chat_api=False, **kwargs):
        url = f"{self.api_base}"

        kwargs = {**self.kwargs, **kwargs}

        stop = kwargs.get("stop")
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens", 150)
        top_p = kwargs.get("top_p", 0.7)
        top_k = kwargs.get("top_k", 50)
        repetition_penalty = kwargs.get("repetition_penalty", 1)
        if self.apply_tokenizer_chat_template:
            prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        # prompt = f"[INST]{prompt}[/INST]" if self.use_inst_template else prompt

        if use_chat_api:
            url = f"{self.api_base}/chat/completions"
            messages = [
                {"role": "system",
                 "content": "You are a helpful assistant. You must continue the user text directly without *any* additional interjections."},
                {"role": "user", "content": prompt},
            ]
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stop": stop,
            }
        else:
            body = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stop": stop,
            }

        headers = {"Authorization": f"Bearer {self.token}"}

        with self.session.post(url, headers=headers, json=body) as resp:
            resp_json = resp.json()
            # Log the token usage from the Together API response.
            self.log_usage(resp_json)
            if use_chat_api:
                # completions = [resp_json['output'].get('choices', [])[0].get('message', {}).get('content', "")]
                completions = [resp_json.get('choices', [])[0].get('message', {}).get('content', "")]
            else:
                # completions = [resp_json['output'].get('choices', [])[0].get('text', "")]
                completions = [resp_json.get('choices', [])[0].get('text', "")]
            response = {"prompt": prompt, "choices": [{"text": c} for c in completions]}
            return response


---

**Assistant:**

Based on the contents of the `lm.py` file, I can provide information about customizing the LLMs you want to use in this project:

1. The file defines several wrapper classes for different LLM providers, including:
   - OpenAIModel
   - AzureOpenAIModel
   - ClaudeModel
   - VLLMClient
   - OllamaClient
   - TGIClient
   - TogetherClient

2. To customize the LLMs you want to use, you can modify or extend these classes based on your requirements. Each class has its own initialization parameters and methods for making API calls to the respective LLM providers.

3. Here's how you can customize the LLMs:

   a. OpenAI and Azure OpenAI:
   - You can specify the model, API key, and other parameters when initializing the OpenAIModel or AzureOpenAIModel classes.
   - Example:
     ```python
     openai_model = OpenAIModel(model="gpt-3.5-turbo", api_key="your-api-key", temperature=0.7)
     azure_model = AzureOpenAIModel(api_base="your-azure-endpoint", api_version="2023-05-15", model="your-azure-model", api_key="your-azure-api-key")
     ```

   b. Claude (Anthropic):
   - You can specify the model and API key when initializing the ClaudeModel class.
   - Example:
     ```python
     claude_model = ClaudeModel(model="claude-2", api_key="your-anthropic-api-key")
     ```

   c. VLLM:
   - You can specify the model, port, and URL when initializing the VLLMClient class.
   - Example:
     ```python
     vllm_model = VLLMClient(model="your-vllm-model", port=8000, url="http://localhost")
     ```

   d. Ollama:
   - You can specify the model, port, and URL when initializing the OllamaClient class.
   - Example:
     ```python
     ollama_model = OllamaClient(model="your-ollama-model", port=11434, url="http://localhost")
     ```

   e. TGI (Text Generation Inference):
   - You can specify the model, port, and URL when initializing the TGIClient class.
   - Example:
     ```python
     tgi_model = TGIClient(model="your-tgi-model", port=8080, url="http://localhost")
     ```

   f. Together AI:
   - You can specify the model and other parameters when initializing the TogetherClient class.
   - Example:
     ```python
     together_model = TogetherClient(model="your-together-model", apply_tokenizer_chat_template=True)
     ```

4. To use these customized LLMs in your project, you would typically import the appropriate class and create an instance with your desired configuration. Then, you can use this instance to make API calls to the LLM.

5. The file also includes methods for tracking token usage, which can be helpful for monitoring and managing API costs.

To customize the LLMs you want to use, you would modify the code that initializes these models in your project, likely in a configuration file or where the LLM instances are created. You may need to provide API keys, specify different models, or adjust other parameters based on your specific requirements.

If you need to add support for a new LLM provider, you could create a new class following a similar pattern to the existing ones, implementing the necessary methods for initialization and API calls.
[Tool Use: attempt_completion]
Result: To customize the LLMs you want to use in this project, you can modify the `knowledge_storm/lm.py` file. This file contains wrapper classes for various LLM providers, including OpenAI, Azure OpenAI, Claude (Anthropic), VLLM, Ollama, TGI (Text Generation Inference), and Together AI.

Here's a summary of how you can customize the LLMs:

1. Choose the appropriate class for your LLM provider (e.g., OpenAIModel, ClaudeModel, VLLMClient, etc.).

2. Initialize the chosen class with your specific parameters. For example:

   - For OpenAI:
     ```python
     openai_model = OpenAIModel(model="gpt-3.5-turbo", api_key="your-api-key", temperature=0.7)
     ```

   - For Claude:
     ```python
     claude_model = ClaudeModel(model="claude-2", api_key="your-anthropic-api-key")
     ```

   - For VLLM:
     ```python
     vllm_model = VLLMClient(model="your-vllm-model", port=8000, url="http://localhost")
     ```

3. You can adjust parameters such as the model name, API keys, endpoints, and other model-specific settings when initializing these classes.

4. If you need to add support for a new LLM provider, you can create a new class in the `lm.py` file, following the pattern of the existing classes. Implement the necessary methods for initialization and API calls.

5. To use your customized LLMs in the project, you would typically import the appropriate class and create an instance with your desired configuration in the part of your code where you initialize the LLM.

Remember to handle API keys securely, preferably using environment variables or a secure configuration management system.

If you need to make more extensive customizations or add entirely new functionality, you may need to modify the existing classes or create new ones in the `lm.py` file. Always ensure that any changes you make are compatible with the rest of the project's structure and requirements.

---

**User:**

[Tool Result]
The user is not pleased with the results. Use the feedback they provided to successfully complete the task, and then attempt completion again.
User's feedback:
"I'm not sure how I will initialise STORM with a custom model; when can i do this?"

---

**Assistant:**

I apologize for the confusion. Let's look into how STORM can be initialized with a custom model. We'll examine the STORM engine to see where and how models are initialized.
[Tool Use: read_file]
Path: knowledge_storm/storm_wiki/engine.py

---

**User:**

[Tool Result]
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Union, Literal, Optional

import dspy

from .modules.article_generation import StormArticleGenerationModule
from .modules.article_polish import StormArticlePolishingModule
from .modules.callback import BaseCallbackHandler
from .modules.knowledge_curation import StormKnowledgeCurationModule
from .modules.outline_generation import StormOutlineGenerationModule
from .modules.persona_generator import StormPersonaGenerator
from .modules.retriever import StormRetriever
from .modules.storm_dataclass import StormInformationTable, StormArticle
from ..interface import Engine, LMConfigs
from ..lm import OpenAIModel
from ..utils import FileIOHelper, makeStringRed


class STORMWikiLMConfigs(LMConfigs):
    """Configurations for LLM used in different parts of STORM.

    Given that different parts in STORM framework have different complexity, we use different LLM configurations
    to achieve a balance between quality and efficiency. If no specific configuration is provided, we use the default
    setup in the paper.
    """

    def __init__(self):
        self.conv_simulator_lm = None  # LLM used in conversation simulator except for question asking.
        self.question_asker_lm = None  # LLM used in question asking.
        self.outline_gen_lm = None  # LLM used in outline generation.
        self.article_gen_lm = None  # LLM used in article generation.
        self.article_polish_lm = None  # LLM used in article polishing.

    def init_openai_model(
            self,
            openai_api_key: str,
            openai_type: Literal["openai", "azure"],
            api_base: Optional[str] = None,
            api_version: Optional[str] = None,
            temperature: Optional[float] = 1.0,
            top_p: Optional[float] = 0.9
    ):
        """Legacy: Corresponding to the original setup in the NAACL'24 paper."""
        openai_kwargs = {
            'api_key': openai_api_key,
            'api_provider': openai_type,
            'temperature': temperature,
            'top_p': top_p,
            'api_base': None
        }
        if openai_type and openai_type == 'openai':
            self.conv_simulator_lm = OpenAIModel(model='gpt-3.5-turbo-instruct',
                                                 max_tokens=500, **openai_kwargs)
            self.question_asker_lm = OpenAIModel(model='gpt-3.5-turbo',
                                                 max_tokens=500, **openai_kwargs)
            # 1/12/2024: Update gpt-4 to gpt-4-1106-preview. (Currently keep the original setup when using azure.)
            self.outline_gen_lm = OpenAIModel(model='gpt-4-0125-preview',
                                              max_tokens=400, **openai_kwargs)
            self.article_gen_lm = OpenAIModel(model='gpt-4o-2024-05-13',
                                              max_tokens=700, **openai_kwargs)
            self.article_polish_lm = OpenAIModel(model='gpt-4o-2024-05-13',
                                                 max_tokens=4000, **openai_kwargs)
        else:
            logging.warning('No valid OpenAI API provider is provided. Cannot use default LLM configurations.')

    def set_conv_simulator_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.conv_simulator_lm = model

    def set_question_asker_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.question_asker_lm = model

    def set_outline_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.outline_gen_lm = model

    def set_article_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_gen_lm = model

    def set_article_polish_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_polish_lm = model


@dataclass
class STORMWikiRunnerArguments:
    """Arguments for controlling the STORM Wiki pipeline."""
    output_dir: str = field(
        metadata={"help": "Output directory for the results."},
    )
    max_conv_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of questions in conversational question asking."},
    )
    max_perspective: int = field(
        default=3,
        metadata={"help": "Maximum number of perspectives to consider in perspective-guided question asking."},
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of search queries to consider in each turn."},
    )
    disable_perspective: bool = field(
        default=False,
        metadata={"help": "If True, disable perspective-guided question asking."},
    )
    search_top_k: int = field(
        default=3,
        metadata={"help": "Top k search results to consider for each search query."},
    )
    retrieve_top_k: int = field(
        default=3,
        metadata={"help": "Top k collected references for each section title."},
    )
    max_thread_num: int = field(
        default=10,
        metadata={"help": "Maximum number of threads to use. "
                          "Consider reducing it if keep getting 'Exceed rate limit' error when calling LM API."},
    )


class STORMWikiRunner(Engine):
    """STORM Wiki pipeline runner."""

    def __init__(self,
                 args: STORMWikiRunnerArguments,
                 lm_configs: STORMWikiLMConfigs,
                 rm):
        super().__init__(lm_configs=lm_configs)
        self.args = args
        self.lm_configs = lm_configs

        self.retriever = StormRetriever(rm=rm, k=self.args.retrieve_top_k)
        storm_persona_generator = StormPersonaGenerator(self.lm_configs.question_asker_lm)
        self.storm_knowledge_curation_module = StormKnowledgeCurationModule(
            retriever=self.retriever,
            persona_generator=storm_persona_generator,
            conv_simulator_lm=self.lm_configs.conv_simulator_lm,
            question_asker_lm=self.lm_configs.question_asker_lm,
            max_search_queries_per_turn=self.args.max_search_queries_per_turn,
            search_top_k=self.args.search_top_k,
            max_conv_turn=self.args.max_conv_turn,
            max_thread_num=self.args.max_thread_num
        )
        self.storm_outline_generation_module = StormOutlineGenerationModule(
            outline_gen_lm=self.lm_configs.outline_gen_lm
        )
        self.storm_article_generation = StormArticleGenerationModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            retrieve_top_k=self.args.retrieve_top_k,
            max_thread_num=self.args.max_thread_num
        )
        self.storm_article_polishing_module = StormArticlePolishingModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            article_polish_lm=self.lm_configs.article_polish_lm
        )

        self.lm_configs.init_check()
        self.apply_decorators()

    def run_knowledge_curation_module(self,
                                      ground_truth_url: str = "None",
                                      callback_handler: BaseCallbackHandler = None) -> StormInformationTable:

        information_table, conversation_log = self.storm_knowledge_curation_module.research(
            topic=self.topic,
            ground_truth_url=ground_truth_url,
            callback_handler=callback_handler,
            max_perspective=self.args.max_perspective,
            disable_perspective=False,
            return_conversation_log=True
        )

        FileIOHelper.dump_json(conversation_log, os.path.join(self.article_output_dir, 'conversation_log.json'))
        information_table.dump_url_to_info(os.path.join(self.article_output_dir, 'raw_search_results.json'))
        return information_table

    def run_outline_generation_module(self,
                                      information_table: StormInformationTable,
                                      callback_handler: BaseCallbackHandler = None) -> StormArticle:

        outline, draft_outline = self.storm_outline_generation_module.generate_outline(
            topic=self.topic,
            information_table=information_table,
            return_draft_outline=True,
            callback_handler=callback_handler
        )
        outline.dump_outline_to_file(os.path.join(self.article_output_dir, 'storm_gen_outline.txt'))
        draft_outline.dump_outline_to_file(os.path.join(self.article_output_dir, "direct_gen_outline.txt"))
        return outline

    def run_article_generation_module(self,
                                      outline: StormArticle,
                                      information_table=StormInformationTable,
                                      callback_handler: BaseCallbackHandler = None) -> StormArticle:

        draft_article = self.storm_article_generation.generate_article(
            topic=self.topic,
            information_table=information_table,
            article_with_outline=outline,
            callback_handler=callback_handler
        )
        draft_article.dump_article_as_plain_text(os.path.join(self.article_output_dir, 'storm_gen_article.txt'))
        draft_article.dump_reference_to_file(os.path.join(self.article_output_dir, 'url_to_info.json'))
        return draft_article

    def run_article_polishing_module(self,
                                     draft_article: StormArticle,
                                     remove_duplicate: bool = False) -> StormArticle:

        polished_article = self.storm_article_polishing_module.polish_article(
            topic=self.topic,
            draft_article=draft_article,
            remove_duplicate=remove_duplicate
        )
        FileIOHelper.write_str(polished_article.to_string(),
                               os.path.join(self.article_output_dir, 'storm_gen_article_polished.txt'))
        return polished_article

    def post_run(self):
        """
        Post-run operations, including:
        1. Dumping the run configuration.
        2. Dumping the LLM call history.
        """
        config_log = self.lm_configs.log()
        FileIOHelper.dump_json(config_log, os.path.join(self.article_output_dir, 'run_config.json'))

        llm_call_history = self.lm_configs.collect_and_reset_lm_history()
        with open(os.path.join(self.article_output_dir, 'llm_call_history.jsonl'), 'w') as f:
            for call in llm_call_history:
                if 'kwargs' in call:
                    call.pop('kwargs')  # All kwargs are dumped together to run_config.json.
                f.write(json.dumps(call) + '\n')

    def _load_information_table_from_local_fs(self, information_table_local_path):
        assert os.path.exists(information_table_local_path), makeStringRed(
            f"{information_table_local_path} not exists. Please set --do-research argument to prepare the conversation_log.json for this topic.")
        return StormInformationTable.from_conversation_log_file(information_table_local_path)

    def _load_outline_from_local_fs(self, topic, outline_local_path):
        assert os.path.exists(outline_local_path), makeStringRed(
            f"{outline_local_path} not exists. Please set --do-generate-outline argument to prepare the storm_gen_outline.txt for this topic.")
        return StormArticle.from_outline_file(topic=topic, file_path=outline_local_path)

    def _load_draft_article_from_local_fs(self, topic, draft_article_path, url_to_info_path):
        assert os.path.exists(draft_article_path), makeStringRed(
            f"{draft_article_path} not exists. Please set --do-generate-article argument to prepare the storm_gen_article.txt for this topic.")
        assert os.path.exists(url_to_info_path), makeStringRed(
            f"{url_to_info_path} not exists. Please set --do-generate-article argument to prepare the url_to_info.json for this topic.")
        article_text = FileIOHelper.load_str(draft_article_path)
        references = FileIOHelper.load_json(url_to_info_path)
        return StormArticle.from_string(topic_name=topic, article_text=article_text, references=references)

    def run(self,
            topic: str,
            ground_truth_url: str = '',
            do_research: bool = True,
            do_generate_outline: bool = True,
            do_generate_article: bool = True,
            do_polish_article: bool = True,
            remove_duplicate: bool = False,
            callback_handler: BaseCallbackHandler = BaseCallbackHandler()):
        """
        Run the STORM pipeline.

        Args:
            topic: The topic to research.
            ground_truth_url: A ground truth URL including a curated article about the topic. The URL will be excluded.
            do_research: If True, research the topic through information-seeking conversation;
             if False, expect conversation_log.json and raw_search_results.json to exist in the output directory.
            do_generate_outline: If True, generate an outline for the topic;
             if False, expect storm_gen_outline.txt to exist in the output directory.
            do_generate_article: If True, generate a curated article for the topic;
             if False, expect storm_gen_article.txt to exist in the output directory.
            do_polish_article: If True, polish the article by adding a summarization section and (optionally) removing
             duplicated content.
            remove_duplicate: If True, remove duplicated content.
            callback_handler: A callback handler to handle the intermediate results.
        """
        assert do_research or do_generate_outline or do_generate_article or do_polish_article, \
            makeStringRed(
                "No action is specified. Please set at least one of --do-research, --do-generate-outline, --do-generate-article, --do-polish-article")

        self.topic = topic
        self.article_dir_name = topic.replace(' ', '_').replace('/', '_')
        self.article_output_dir = os.path.join(self.args.output_dir, self.article_dir_name)
        os.makedirs(self.article_output_dir, exist_ok=True)

        # research module
        information_table: StormInformationTable = None
        if do_research:
            information_table = self.run_knowledge_curation_module(ground_truth_url=ground_truth_url,
                                                                   callback_handler=callback_handler)
        # outline generation module
        outline: StormArticle = None
        if do_generate_outline:
            # load information table if it's not initialized
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, 'conversation_log.json'))
            outline = self.run_outline_generation_module(information_table=information_table,
                                                         callback_handler=callback_handler)

        # article generation module
        draft_article: StormArticle = None
        if do_generate_article:
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, 'conversation_log.json'))
            if outline is None:
                outline = self._load_outline_from_local_fs(topic=topic,
                                                           outline_local_path=os.path.join(self.article_output_dir,
                                                                                           'storm_gen_outline.txt'))
            draft_article = self.run_article_generation_module(outline=outline,
                                                               information_table=information_table,
                                                               callback_handler=callback_handler)

        # article polishing module
        if do_polish_article:
            if draft_article is None:
                draft_article_path = os.path.join(self.article_output_dir, 'storm_gen_article.txt')
                url_to_info_path = os.path.join(self.article_output_dir, 'url_to_info.json')
                draft_article = self._load_draft_article_from_local_fs(topic=topic,
                                                                       draft_article_path=draft_article_path,
                                                                       url_to_info_path=url_to_info_path)
            self.run_article_polishing_module(draft_article=draft_article, remove_duplicate=remove_duplicate)


---

**Assistant:**

Thank you for providing the `engine.py` file. Now I can give you a more accurate answer about how to initialize STORM with a custom model.

To customize the LLMs used in STORM, you can modify the `STORMWikiLMConfigs` class in the `engine.py` file. Here's how you can initialize STORM with custom models:

1. First, create an instance of `STORMWikiLMConfigs`:

```python
lm_configs = STORMWikiLMConfigs()
```

2. Then, set the custom models for different parts of STORM using the provided setter methods:

```python
from your_custom_model_module import YourCustomModel

# Create instances of your custom models
custom_model1 = YourCustomModel(model="your-model-name", **your_model_params)
custom_model2 = YourCustomModel(model="another-model-name", **other_model_params)

# Set the custom models for different parts of STORM
lm_configs.set_conv_simulator_lm(custom_model1)
lm_configs.set_question_asker_lm(custom_model1)
lm_configs.set_outline_gen_lm(custom_model2)
lm_configs.set_article_gen_lm(custom_model2)
lm_configs.set_article_polish_lm(custom_model2)
```

3. Create an instance of `STORMWikiRunnerArguments` with your desired settings:

```python
args = STORMWikiRunnerArguments(
    output_dir="your/output/directory",
    max_conv_turn=3,
    max_perspective=3,
    max_search_queries_per_turn=3,
    search_top_k=3,
    retrieve_top_k=3,
    max_thread_num=10
)
```

4. Initialize the STORM runner with your custom configurations:

```python
from your_retriever_module import YourRetrieverModel

rm = YourRetrieverModel(**retriever_params)
storm_runner = STORMWikiRunner(args=args, lm_configs=lm_configs, rm=rm)
```

5. Now you can run the STORM pipeline with your custom models:

```python
storm_runner.run(
    topic="Your research topic",
    ground_truth_url="",
    do_research=True,
    do_generate_outline=True,
    do_generate_article=True,
    do_polish_article=True
)
```

By following these steps, you can initialize STORM with custom models for different parts of the pipeline. Make sure that your custom model classes are compatible with the expected interface (they should be instances of `dspy.dsp.LM` or `dspy.dsp.HFModel`).

Remember to import your custom model classes and any necessary modules at the beginning of your script.
[Tool Use: attempt_completion]
Result: To initialize STORM with custom models, you can follow these steps:

1. Create an instance of `STORMWikiLMConfigs`:

```python
from knowledge_storm.storm_wiki.engine import STORMWikiLMConfigs, STORMWikiRunnerArguments, STORMWikiRunner

lm_configs = STORMWikiLMConfigs()
```

2. Set custom models for different parts of STORM:

```python
from your_custom_model_module import YourCustomModel

# Create instances of your custom models
custom_model1 = YourCustomModel(model="your-model-name", **your_model_params)
custom_model2 = YourCustomModel(model="another-model-name", **other_model_params)

# Set the custom models for different parts of STORM
lm_configs.set_conv_simulator_lm(custom_model1)
lm_configs.set_question_asker_lm(custom_model1)
lm_configs.set_outline_gen_lm(custom_model2)
lm_configs.set_article_gen_lm(custom_model2)
lm_configs.set_article_polish_lm(custom_model2)
```

3. Create an instance of `STORMWikiRunnerArguments`:

```python
args = STORMWikiRunnerArguments(
    output_dir="your/output/directory",
    max_conv_turn=3,
    max_perspective=3,
    max_search_queries_per_turn=3,
    search_top_k=3,
    retrieve_top_k=3,
    max_thread_num=10
)
```

4. Initialize the STORM runner:

```python
from your_retriever_module import YourRetrieverModel

rm = YourRetrieverModel(**retriever_params)
storm_runner = STORMWikiRunner(args=args, lm_configs=lm_configs, rm=rm)
```

5. Run the STORM pipeline:

```python
storm_runner.run(
    topic="Your research topic",
    ground_truth_url="",
    do_research=True,
    do_generate_outline=True,
    do_generate_article=True,
    do_polish_article=True
)
```

Make sure your custom model classes are compatible with `dspy.dsp.LM` or `dspy.dsp.HFModel`. Import all necessary custom modules at the beginning of your script.

This approach allows you to customize the LLMs used in different parts of the STORM pipeline, giving you flexibility in choosing models that best suit your needs for each task.

