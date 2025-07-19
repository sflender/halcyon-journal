#!/usr/bin/env python3
"""
Local-Only Interactive Journaling App
A privacy-focused journaling application that processes everything locally.
"""

import os
import json
import datetime
from pathlib import Path
from typing import List, Dict, Optional
import ollama
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import typer

console = Console()

class JournalApp:
    def __init__(self):
        self.journal_dir = Path.home() / ".halcyon_journal"
        self.journal_dir.mkdir(exist_ok=True)
        self.embeddings_file = self.journal_dir / "embeddings.json"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ollama_client = ollama.Client()
        
        # Ensure Ollama is running and has a suitable model
        self._setup_ollama()
        
    def _setup_ollama(self):
        """Setup Ollama with a suitable model for local processing."""
        try:
            # Check if we have a suitable model
            models = self.ollama_client.list()
            model_names = [model['name'] for model in models['models']]
            
            # Use llama2 if available, otherwise try to pull it
            if 'llama2' not in model_names:
                console.print("[yellow]Setting up local LLM... This may take a few minutes on first run.[/yellow]")
                self.ollama_client.pull('llama2')
            
            self.model_name = 'llama2'
            console.print("[green]✓ Local LLM ready[/green]")
            
        except Exception as e:
            console.print(f"[red]Error setting up Ollama: {e}[/red]")
            console.print("[yellow]Please ensure Ollama is installed and running: https://ollama.ai[/yellow]")
            raise
    
    def _get_today_filename(self) -> str:
        """Get today's journal filename."""
        today = datetime.date.today()
        return f"{today.strftime('%Y-%m-%d')}.md"
    
    def _load_embeddings(self) -> Dict[str, List[float]]:
        """Load existing embeddings from file."""
        if self.embeddings_file.exists():
            with open(self.embeddings_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_embeddings(self, embeddings: Dict[str, List[float]]):
        """Save embeddings to file."""
        with open(self.embeddings_file, 'w') as f:
            json.dump(embeddings, f)
    
    def _get_entry_embedding(self, content: str) -> List[float]:
        """Get embedding for journal entry content."""
        return self.model.encode(content).tolist()
    
    def _find_similar_entries(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find similar journal entries using semantic search."""
        embeddings = self._load_embeddings()
        if not embeddings:
            return []
        
        # Get query embedding
        query_embedding = self._get_entry_embedding(query)
        
        # Calculate similarities
        similarities = []
        for filename, embedding in embeddings.items():
            similarity = cosine_similarity(
                [query_embedding], [embedding]
            )[0][0]
            similarities.append((filename, similarity))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Load the actual entries
        similar_entries = []
        for filename, similarity in similarities[:top_k]:
            entry_path = self.journal_dir / filename
            if entry_path.exists():
                with open(entry_path, 'r') as f:
                    content = f.read()
                    similar_entries.append({
                        'filename': filename,
                        'content': content,
                        'similarity': similarity
                    })
        
        return similar_entries
    
    def _get_therapist_response(self, current_entry: str, similar_entries: List[Dict]) -> str:
        """Generate a therapist-like response using local LLM."""
        # Build context from similar entries
        context = ""
        if similar_entries:
            context = "\n\nRelated past entries:\n"
            for entry in similar_entries:
                context += f"\n--- {entry['filename']} ---\n{entry['content'][:500]}...\n"
        
        prompt = f"""You are a caring, empathetic therapist. The user has written a journal entry about their day. 
        Please provide a thoughtful, supportive response that:
        1. Acknowledges their feelings and experiences
        2. Offers gentle insights or observations
        3. Asks a thoughtful follow-up question
        4. Maintains a warm, professional tone
        
        Current journal entry:
        {current_entry}
        
        {context}
        
        Your response:"""
        
        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 300
                }
            )
            return response['response']
        except Exception as e:
            return f"I'm here to listen and support you. Your journal entry shows thoughtful reflection. How are you feeling about what you've written today?"
    
    def journal_mode(self):
        """Start a new daily journal entry."""
        console.print(Panel.fit(
            "[bold blue]📝 Journal Mode[/bold blue]\nWrite about your day...",
            border_style="blue"
        ))
        
        today_file = self.journal_dir / self._get_today_filename()
        
        # Check if today's entry already exists
        if today_file.exists():
            with open(today_file, 'r') as f:
                existing_content = f.read()
            console.print(f"[yellow]Today's entry already exists. Current content:[/yellow]")
            console.print(Markdown(existing_content))
            
            if not Confirm.ask("Would you like to add to today's entry?"):
                return
        
        # Get journal entry
        console.print("\n[green]Start writing (press Enter twice to finish):[/green]")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        
        content = "\n".join(lines[:-1])  # Remove the last empty line
        
        if not content.strip():
            console.print("[yellow]No content entered. Exiting journal mode.[/yellow]")
            return
        
        # Save the entry
        with open(today_file, 'a') as f:
            if today_file.exists() and os.path.getsize(today_file) > 0:
                f.write("\n\n---\n\n")  # Separator for additional entries
            f.write(content)
        
        # Update embeddings
        embeddings = self._load_embeddings()
        embeddings[self._get_today_filename()] = self._get_entry_embedding(content)
        self._save_embeddings(embeddings)
        
        console.print(f"\n[green]✓ Journal entry saved to {today_file}[/green]")
        
        # Find similar entries
        similar_entries = self._find_similar_entries(content)
        
        # Generate therapist response
        console.print("\n[bold blue]🤗 Your Personal Reflection:[/bold blue]")
        therapist_response = self._get_therapist_response(content, similar_entries)
        console.print(Panel(Markdown(therapist_response), border_style="green"))
        
        # Show similar entries
        if similar_entries:
            console.print("\n[bold blue]📚 Related Past Entries:[/bold blue]")
            for entry in similar_entries:
                console.print(Panel(
                    f"[bold]{entry['filename']}[/bold] (similarity: {entry['similarity']:.2f})\n\n{entry['content'][:200]}...",
                    border_style="yellow"
                ))
    
    def reflect_mode(self):
        """Let user ask questions about their journal."""
        console.print(Panel.fit(
            "[bold purple]🔍 Reflect Mode[/bold purple]\nAsk questions about your journal...",
            border_style="purple"
        ))
        
        # Check if there are any journal entries
        journal_files = list(self.journal_dir.glob("*.md"))
        if not journal_files:
            console.print("[yellow]No journal entries found. Start journaling first![/yellow]")
            return
        
        while True:
            query = Prompt.ask("\n[green]What would you like to know about your journal?[/green] (or 'quit' to exit)")
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query.strip():
                continue
            
            # Find relevant entries
            similar_entries = self._find_similar_entries(query, top_k=5)
            
            if not similar_entries:
                console.print("[yellow]No relevant entries found for your query.[/yellow]")
                continue
            
            # Generate response using local LLM
            context = "\n\n".join([
                f"Entry from {entry['filename']}:\n{entry['content']}"
                for entry in similar_entries
            ])
            
            prompt = f"""You are a helpful assistant analyzing someone's personal journal. 
            The user asked: "{query}"
            
            Based on these journal entries, provide a thoughtful, insightful response:
            
            {context}
            
            Your response:"""
            
            try:
                response = self.ollama_client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'max_tokens': 400
                    }
                )
                
                console.print(Panel(
                    Markdown(response['response']),
                    title="🤔 Reflection",
                    border_style="purple"
                ))
                
            except Exception as e:
                console.print(f"[red]Error generating response: {e}[/red]")
    
    def run(self):
        """Main application loop."""
        console.print(Panel.fit(
            "[bold]🌟 Halcyon Journal[/bold]\nYour private, local journaling companion",
            border_style="blue"
        ))
        
        while True:
            console.print("\n[bold]Choose a mode:[/bold]")
            console.print("1. 📝 Journal - Write about your day")
            console.print("2. 🔍 Reflect - Ask questions about your journal")
            console.print("3. 🚪 Exit")
            
            choice = Prompt.ask("Select an option", choices=["1", "2", "3"])
            
            if choice == "1":
                self.journal_mode()
            elif choice == "2":
                self.reflect_mode()
            elif choice == "3":
                console.print("[green]Thank you for journaling with Halcyon! 🌟[/green]")
                break

def main():
    """Main entry point."""
    try:
        app = JournalApp()
        app.run()
    except KeyboardInterrupt:
        console.print("\n[green]Goodbye! 🌟[/green]")
    except Exception as e:
        console.print(f"[red]An error occurred: {e}[/red]")
        console.print("[yellow]Please ensure Ollama is installed and running.[/yellow]")

if __name__ == "__main__":
    main() 