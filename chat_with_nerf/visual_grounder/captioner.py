import torch
from attrs import Factory, define
from PIL import Image
from chat_with_nerf.settings import Settings
from chat_with_nerf.visual_grounder.image_ref import ImageRef
from llava.conversation import conv_templates, SeparatorStyle
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria
from typing import List, Optional, Any


@define
class BaseCaptioner:
    model: torch.nn.Module
    """Base model for image processing."""
    vis_processors: dict
    """Preprocessors for visual inputs."""
    images: List[ImageRef] = Factory(list)

    def load_images(self, selected: List[ImageRef]) -> None:
        """Loads images from the provided list of ImageRefs."""
        # load sample image
        for image_ref in selected:
            raw_image = Image.open(image_ref.rgb_address).convert("RGB")
            # raw_image.resize((596, 437))
            image_ref.raw_image = raw_image
        self.images = selected

    def process_image(self, image_path: str) -> torch.Tensor:
        """Processes an image and returns it as a tensor."""
        raw_image = Image.open(image_path).convert("RGB")
        # raw_image.resize((596, 437))
        return self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.model.device)

    def set_positive_words(self) -> None:
        """Sets the positive words for the captioner."""
        pass

    def caption(self) -> None:
        """Generates captions for the images."""
        pass


@define
class Blip2Captioner(BaseCaptioner):
    positive_words: str = "computer"

    def set_positive_words(self, new_positive_words):
        self.positive_words = new_positive_words

    def caption(self) -> dict[str, str]:
        """_summary_

        :return: a dictionary of image path and its corresponding caption
        :rtype: dict[str, str]
        """
        # TODO: batch it.
        # prepare the image
        result: dict[str, str] = {}  # key: rgb_address, value: caption
        for image_ref in self.images:
            image = self.process_image(image_ref.raw_image)
            question = (
                "Describe the shape and material of the" 
                + self.positive_words
                + ", if there is one."
            )
            answer = self.model.generate({"image": image, "prompt": question})[0]  # type: ignore
            result[image_ref.rgb_address] = answer

        return result


@define
class LLaVaCaptioner(BaseCaptioner):
    tokenizer: Optional[Any] = None
    mm_use_im_start_end: bool = True
    image_token_len: int = 512
    positive_words: str = "computer"

    def set_positive_words(self, new_positive_words):
        self.positive_words = new_positive_words

    def caption(self) -> dict[str, str]:
        """_summary_

        :return: a dictionary of image path and its corresponding caption
        :rtype: dict[str, str]
        """
        qs = (
            "Describe the shape and material of the" 
            + self.positive_words
            + ", if there is one."
        )

        if self.mm_use_im_start_end:
            qs = (
                qs 
                + '\n' 
                + Settings.DEFAULT_IM_START_TOKEN 
                + Settings.DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len 
                + Settings.DEFAULT_IM_END_TOKEN
            )
        else:
            qs = qs + '\n' + Settings.DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len
        
        conv_mode = "multimodal"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        input_token_len = input_ids.shape[1]
        # prepare the image
        result: dict[str, str] = {}  # key: rgb_address, value: caption
        for image_ref in self.images:
            image = self.process_image(image_ref.raw_image)
            image_tensor = self.vis_processors["image_processor"].preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria]
                )
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids' \
                        + 'are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:],
                skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            print(outputs)
            result[image_ref.rgb_address] = outputs
        return result
