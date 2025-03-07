package com.sopra.jpa.entities;

import java.io.Serializable;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.Table;

@Entity
//@Table(name="planta") //Relaciona la entidad con la tabla en la base de datos
public class Planta implements Serializable {

	//definición de propiedades
	@Id
	@Column(name="id")
	private int id;
	
	@Column(name="nombre")
	private String nombre;
	
	@Column(name="tipo")
	private String tipo;
	
	//Constructores
	public Planta() {
		
	}

	public Planta( int id, String nombre, String tipo) {
		this.id = id;
		this.nombre = nombre;
		this.tipo = tipo;
	}
	
	//Métodos get y set de las propiedades
	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public String getNombre() {
		return nombre;
	}

	public void setNombre(String nombre) {
		this.nombre = nombre;
	}

	public String getTipo() {
		return tipo;
	}

	public void setTipo(String tipo) {
		this.tipo = tipo;
	}
	
	
	@Override
	public String toString() {
		return "Planta [id=" + id + ", nombre=" + nombre + ", tipo=" + tipo + "]";
	}
}
